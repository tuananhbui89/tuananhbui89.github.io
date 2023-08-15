---
title: 'On Reading: Anti-DreamBooth: Protecting users from personalized text-to-image synthesis'
date: 2023-08-05
permalink: /posts/2023/08/papers/antidreambooth/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
  - Trustworthy machine learning
---
<br>

- [About the paper](#about-the-paper)
- [How to preprocess the data](#how-to-preprocess-the-data)
    - [Difference in prompting process between "Textual Inversion" and "Dreambooth" projects](#difference-in-prompting-process-between-textual-inversion-and-dreambooth-projects)
- [PGD attack](#pgd-attack)


About the paper 
=====
- Published at ICCV 2023 
- Authors: VinAI 
- Main idea: Generate invisible perturbation to add to the personal images before uploading to the internet. To prevent adversary from using the uploaded images to train a personalized model (specific to Dreambooth method) to generate harmful images (e.g., nude images) of the person. 

How to preprocess the data
=====
It is worth noting that in adversarial examples, the perturbation is added to the post-processed image while in this project the perturbation should be added to the pre-processed image and robust to the pre-processing step. However, in the current implementation, the perturbation is added to the post-processed image.

- It first call the `load_data` function that read PIL image and apply some transformations (e.g., resize, crop, normalize) and return a tensor (shape = [N, H, W, C], channel last format???). Ref: [Line 360](https://github.com/VinAIResearch/Anti-DreamBooth/blob/0d1ed6ff4766a876e65753f8c00fad8bf48f37c6/attacks/aspl.py#L360)  

<!-- Insert code block -->
```python
def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = torch.stack(images)
    return images
```

- It then call the `DreamBoothDatasetFromTensor` class with input argument `instance_images_tensor` which is the tensor returned from the `load_data` function. In this class, when `__getitem__` function is called, the data will be associated with a corresponding textual prompt. There is **no transformation applied** in this class. Ref: [Line 31](https://github.com/VinAIResearch/Anti-DreamBooth/blob/0d1ed6ff4766a876e65753f8c00fad8bf48f37c6/attacks/aspl.py#L31) 

```python
    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example
```

The overall pipeline is as the snippet code below: 
- Load the data from the `instance_data_dir_for_adversarial` directory. Output is a tensor of shape [N, H, W, C]? should it be [N, C, H, W]?
```
    # Load data from the instance data directory 
    # output: tensor of shape [N, H, W, C]? should it be [N, C, H, W]?
    perturbed_data = load_data(
        args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
```
- Clone the current model to avoid the in-place operation
- Train the model with the clean data 
- Learn the perturbation with the updated model f_sur. Input is the entire data tensor not just a batch! Output is the new perturbed data tensor.
- Restore the model and train with perturbed data

```python
    f = [unet, text_encoder]
    for i in range(args.max_train_steps):
        # Clone the current model to avoid the in-place operation
        f_sur = copy.deepcopy(f)
        # Train the model with the clean data
        f_sur = train_one_epoch(
            args,
            f_sur,
            tokenizer,
            noise_scheduler,
            vae,
            clean_data,
            args.max_f_train_steps,
        )
        # Learn the perturbation with the updated model f_sur 
        perturbed_data = pgd_attack(
            args,
            f_sur,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            original_data,
            target_latent_tensor,
            args.max_adv_train_steps,
        )
        # Restore the model and train with perturbed data 
        f = train_one_epoch(
            args,
            f,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            args.max_f_train_steps,
        )
```

Inside the `train_one_epoch` function, the `DreamBoothDatasetFromTensor` class is called to associate the data (i.e., perturbed data) with the corresponding textual prompt.

```python
    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )
```

At the end, the perturbed image is saved in the `instance_data_dir_for_adversarial` directory. 

Some notes: 
- In the original Dreambooth project, the data is loaded from the `DataLoader` class and is shuffled. However, in the Anti-Dreambooth project, the data is loaded from the `DreamBoothDatasetFromTensor` class and is not shuffled. Ref: [Line 1061](https://github.com/huggingface/diffusers/blob/ea1fcc28a458739771f5112767f70d281511d2a2/examples/dreambooth/train_dreambooth.py#L1061)
- The reason for the above modification is that the author want to change on the fly the perturbed data after each epoch, which will be harder in control if using `DataLoader` class.

### Difference in prompting process between "Textual Inversion" and "Dreambooth" projects

In Dreambooth, there is an argument `instance_prompt` which is used as a neural prompt to associate with the given images. For example, the default value is `a photo of sks dog`, where `sks` is the unique identifier to specify the learned concept. The `instance_prompt` is then tokenized by the tokenizer and the token ids are used to specify the position in the embedding matrix to be updated (corresponding to the specific token).

```python
    # In the DreamBoothDataset class
    if self.encoder_hidden_states is not None:
        example["instance_prompt_ids"] = self.encoder_hidden_states
    else:
        text_inputs = tokenize_prompt(
            self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask
```

So the difference between the two projects is that:
- In Dreambooth, only one neural prompt is used, while in Textual Inversion, there is a list of neural prompts
- In Textual Inversion, it is important to specify the `placeholder_token` to reuse the same token in other prompts, while in Dreambooth, the identifier (i.e., `sks`) is used to specify the position in the embedding matrix to be updated (corresponding to the specific token). In inferencce, a prompt with the same identifier will be used to generate images, for example, `a photo of sks dog in the beach`. So to me, the whole prompt in Dreambooth is like a placeholder token in Textual Inversion. However, in this case, how the output looks like if we use a prompt that not contains the whole `instance_prompt`? For example, `a sks dog walking on the beach`?

PGD attack 
=====

The PGD attack is implemented in the `pgd_attack` function. The input is the perturbed data tensor and the output is the new perturbed data tensor. 

Some notes: 
- weight type is `torch.bfloat16` instead of `torch.float32`
- unet, vae, text_encoder are in `train` mode (because they were set in `train_one_epoch` function)
- Learn for the entire data tensor not just a batch
- The whole process is quite similar to the standard PGD attack without the random initialization.

```python
    # Create a copy of data and set requires_grad to True
    perturbed_images = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)

    # Repeat the input_ids to match the batch size
    input_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)

    # Loop over the number of steps
    for step in range(num_steps):

        # Reset requires_grad to True because it was set to False in the last step
        perturbed_images.requires_grad = True
        latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        unet.zero_grad()
        text_encoder.zero_grad()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # target-shift loss
        if target_tensor is not None:
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1],
                        timesteps[idx : idx + 1],
                        noisy_latents[idx : idx + 1],
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise, timesteps - 1)
            loss = loss - F.mse_loss(xtm1_pred, xtm1_target)

        loss.backward()

        alpha = args.pgd_alpha
        eps = args.pgd_eps

        # Project to valid range
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
        perturbed_images = torch.clamp(original_images + eta, min=-1, max=+1).detach_()
        print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
    return perturbed_images
```