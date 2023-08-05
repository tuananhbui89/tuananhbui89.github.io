---
title: 'On Reading: Anti-Dreambooth'
date: 2023-08-05
permalink: /posts/2023/08/papers/antidreambooth/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
  - TML
---
<br>

- [About the paper](#about-the-paper)
- [How to preprocess the data](#how-to-preprocess-the-data)


About the paper 
=====
- Published at ICCV 2023 
- Authors: VinAI 
- Main idea: Generate invisible perturbation to add to the personal images before uploading to the internet. To prevent adversary from using the uploaded images to train a personalized model (specific to Dreambooth method) to generate harmful images (e.g., nude images) of the person. 

How to preprocess the data
=====
It is worth noting that in adversarial examples, the perturbation is added to the post-processed image while in this project the perturbation should be added to the pre-processed image and robust to the pre-processing step. However, in the current implementation, the perturbation is added to the post-processed image. 

- It first call the `load_data` function that read PIL image and apply some transformations (e.g., resize, crop, normalize) and return a tensor (shape = [N, H, W, C], channel last format???). Ref: https://github.com/VinAIResearch/Anti-DreamBooth/blob/0d1ed6ff4766a876e65753f8c00fad8bf48f37c6/attacks/aspl.py#L360 

<!-- Insert code block -->
```
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

- It then call the `DreamBoothDatasetFromTensor` class with input argument `instance_images_tensor` which is the tensor returned from the `load_data` function. In this class, when `__getitem__` function is called, the data will be associated with a corresponding textual prompt. There is **no transformation applied** in this class. Ref: https://github.com/VinAIResearch/Anti-DreamBooth/blob/0d1ed6ff4766a876e65753f8c00fad8bf48f37c6/attacks/aspl.py#L31

```
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

```
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
```
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
