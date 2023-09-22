---
title: 'On Reading: Diffusion Models Beat GANs on Image Synthesis'
date: 2023-09-14
permalink: /posts/2023/09/papers/conditional-diffusion/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
---
<br>

- [About the paper](#about-the-paper)
- [Understanding Conditional Diffusion Process](#understanding-conditional-diffusion-process)
- [Classifier Guidance](#classifier-guidance)
- [How to implement](#how-to-implement)
  - [How to train the diffusion model](#how-to-train-the-diffusion-model)
  - [How to train the classifier](#how-to-train-the-classifier)
- [References](#references)

About the paper
=====

- Published at NeurIPS 2021
- Affiliations: OpenAI
- One of the very first works on diffusion model. Showing that diffusion model can be used for image synthesis and outperform GANs on FID score. One important contribution of the paper is proposing conditional diffusion process by using gradient from an auxiliary classifier, which is used to sample images from a specific class
- Link to the paper: [https://arxiv.org/pdf/2105.05233.pdf](https://arxiv.org/pdf/2105.05233.pdf)
- Link to the code: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)

Understanding Conditional Diffusion Process
=====

In this section, we will go through the Conditional Diffusion Process introduced in Appendix H of the paper.

We start by defining a conditional Markovian noising process $\hat{q}$ similar to $q$, and assume that $\hat{q}(y \mid x_0)$ is a known and readily available label distribution for each sample.

- $\hat{q}(x_0) = q(x_0)$: the initial distribution of the process is the same as the unconditional process.
- $\hat{q}(y \mid x_0)$ is the label distribution for each sample $x_0$ which is known and readily available.
- $\hat{q}(x_{t+1} \mid x_t, y) = q(x_{t+1} \mid x_t)$: **This is the key point that will later enable us to derive the conditional diffusion process**. This explains that the transition distribution is the same as the unconditional process, i.e., the noise adding in the forward diffusiion process is independent to label $y$. However, this might not neccessary be the case. If using SDE (Stochastic Differential Equation) to model the diffusion process, then the forward diffusion process can be conditioned on $y$. **This can be a future work to explore.**
- $\hat{q}(x_{1:T} \mid x_0, y) = \prod_{t=1}^T \hat{q}(x_t \mid x_{t-1}, y)$

From the above definition, we can derive the following properties:

- $\hat{q}(x_{t+1} \mid x_t, y) = \hat{q}(x_{t+1} \mid x_t) = q(x_{t+1} \mid x_t)$ the forward process conditioned on $y$ is the same as the unconditional forward process.
- $\hat{q}(y \mid x_t, x_{t+1}) = \hat{q}(y \mid x_t)$: the label distribution is independent of the next sample $x_{t+1}$.
- $\hat{q}(y \mid x_t, x_{t+1}) \neq \hat{q}(y \mid x_{t+1})$: Need confirmation on this. But if this is true, then it means that $\hat{q}(y \mid x_t) \neq \hat{q}(y \mid x_{t+1})$ or the label distribution has changed after adding noise at each step. Then we cannot use the same classifier to approximate the label distribution at each step. **However, in the paper, the authors still use the same classifier!!!**. One possible idea is that we can consider a classifier that is conditioned to time step $t$.

Based on the above properties, we now can derive the conditional reverse process as follows:

$$\hat{q}(x_{t} \mid x_{t+1}, y) = \frac{\hat{q}(x_{t} \mid x_{t+1}) \hat{q}(y \mid x_{t}, x_{t+1})}{\hat{q}(y \mid x_{t+1})}$$

$$\hat{q}(x_{t} \mid x_{t+1}, y) = \frac{q(x_{t} \mid x_{t+1}) \hat{q}(y \mid x_{t})}{\hat{q}(y \mid x_{t+1})}$$

The term $\hat{q}(y \mid x_{t+1})$ is considered as constant w.r.t. $x_t$. So $x_t$ can be sampled from the above distribution, where $\hat{q}(y \mid x_{t})$ is approximated by an auxiliary classifier, which is trained to predict the label $y$ from the sample $x_t$. And $q(x_{t} \mid x_{t+1})$ is the reverse process of the unconditional diffusion process which has been trained.

Classifier Guidance
=====

After understanding the conditional diffusion process, we now go through the classifier guidance to see how to use the classifier to guide the sampling process.
In the paper, the authors proposed two sampling approaches: 

- **Conditional Reverse Noising Process**: which factorizes the conditional transition $p_{\theta, \phi}(x_t \mid x_{t+1}, y) = Z p_\theta(x_t \mid x_{t+1} p_\phi (y \mid x_t))$. This can be approximated by a Gaussian similar to the unconditional reverse process, but with its mean shifted by $\Sigma g$
- **Conditional Sampling for DDIM**: which can be applied for deterministic sampling methods like DDIM. This can be done by using the conditioning trick adapted from Song et al. (2021). 

![Two sampling methods](https://raw.githubusercontent.com/tuananhbui89/tuananhbui89.github.io/master/images/2309/classifier_guidance/two_sampling_methods.png)

How to implement
=====

Link to the original implementation: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)

Minimal code to implement the classifier guidance diffusion as follows, which is based on the code from file [`classifier_sample.py`](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py)


```python

def main():
    # create unet and scheduler of the diffusion model
    model, diffusion = create_model_and_diffusion()

    # load the pretrained unet 
    model.load_state_dict()
    model.eval()

    # create classifier which is Unet's encoder with linear layer on top
    classifier = create_classifier()

    # load the pretrained classifier
    classifier.load_state_dict()
    classifier.eval()

    # define the gradient of the classifier w.r.t. the input as guidance for sampling 
    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale   
    

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # main loop 
    while gothrough_all_images:
        # random target classes
        classes = torch.randint()

        # calling sample function with the classifier guidance 
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn, # classifier guidance
            device=dist_util.dev(),
        )

        # save the output 
```

The crucial part of the above code is the `cond_fn` function which defines the gradient of the classifier w.r.t. the input as guidance for sampling. Another important part is the `diffusion.p_sample_loop` or `diffusion.ddim_sample_loop` which will use the classifier guidance to sample images from the diffusion model.

The diffusion model with these above sampling methods can be found in the file [`script_util.py`](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/script_util.py#L74) and the sampling methods are defined in the file [`gaussian_diffusion.py`](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)

The Algorithm 1 (Conditional Reverse Noising Process, i.e., `p_sample_loop`) can be implemented as follows:

```python

# one step of the sampling process
def p_sample(self, model, x, t, clip_denoised, denoised_fn, cond_fn, model_kwargs):

    # sample from the unconditional reverse process
    # the output includes "mean" and "log_variance" of the Gaussian distribution
    # the output also includes "pred_xstart"
    out = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)

    # Shift the mean by the gradient of the classifier w.r.t. the input
    # equation: new_mean = mean + sigma * g 
    # where sigma is the standard deviation of the Gaussian distribution, i.e., out["varaince"]
    if cond_fn is not None:
        out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs)
    
    # create nonzero mask
    nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

    # sample from the shifted Gaussian distribution
    sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(out["mean"])
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}

# the progressive sampling loop from T to 0, where the $x_t$ will be used to sample $x_{t-1}$

def p_sample_loop_progressive():

    ...

    for i in indices:
        t = th.tensor([i] * shape[0], device=device)
        with th.no_grad():
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            yield out
            img = out["sample"]

```

The Algorithm 2 (Conditional Sampling for DDIM, i.e., `ddim_sample_loop`) can be implemented as below. As described in the paper, the stochastic process can be controlled by the parameter `eta`. When `eta=0`, the sampling process is truly deterministic, while `eta > 0`, the sampling process is stochastic.

<!-- However, one point that I am still not understand is that the DDIM is the deterministic sampling method, however, in the code, at the end, we still sample from a Gaussian distribution with the mean as calculated in Algorithm 2. The sigma is controlled by additional parameter `eta`, where `eta=0` means truly deterministic sampling. -->


```python

# one step of the sampling process
def ddim_sample(self, model, x, t, clip_denoised, denoised_fn, cond_fn, model_kwargs, eta):

    # sample from the unconditional reverse process
    # the output includes "mean" and "log_variance" of the Gaussian distribution
    # the output also includes "pred_xstart"
    out = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)

    # calculate score 
    if cond_fn is not None:
        out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)    
    
    # calculate epsilon_t 
    eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

    # calculate alpha_bar_t and alpha_bar_prev and sigma 
    sigma = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
    )    

    # calculate x_{t-1} as in Algorithm 2
    mean_pred = (
        out["pred_xstart"] * th.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )

    # Still random sample from a Gaussian distribution 
    # but the mean is calculated as above
    sample = mean_pred + nonzero_mask * sigma * torch.randn_like(mean_pred)
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}    
```

The main component of the above code is the unconditional reverse process `p_mean_variance` which is defined as follows in the file `gaussian_diffusion.py`. It is worth noting that this function not only returns the $x_{t-1}$ but also the prediction of the initial $x_0$, i.e., `pred_xstart`. 


```python

def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):

    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                    as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
                - 'mean': the model mean output.
                - 'variance': the model variance output.
                - 'log_variance': the log of 'variance'.
                - 'pred_xstart': the prediction for x_0.
    """

    model_output = model(x, self._scale_timesteps(t), **model_kwargs)

    # really long process 
    ... 


    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }
```


## How to train the diffusion model

Training the diffusion model in this project is similar as in the DDPM or DDIM papers. Because even using auxiliary classifier, they are trained independently. The minimal code to train the diffusion model is as follows, which is based on the code from file [`image_train.py`](https://github.com/openai/guided-diffusion/blob/main/scripts/image_train.py) and [`train_util.py`](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/train_util.py#L22)


```python

# run loop
def run_loop(self):
    while some_conditions:
        batch, cond = next(self.data)
        # run one step
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
    return None

# forward and backward
def forward_backward(self, batch, cond):
    self.mp_trainer.zero_grad()

    for i in range(0, batch.shape[0], self.microbatch):
        micro, micro_cond = ... 

        # sampling time step t and the weights from a schedule sampler (e.g, uniform))
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
        losses = compute_losses()

        loss = (losses["loss"] * weights).mean()

        self.mp_trainer.backward(loss)
    
    return None

# where the diffusion.training_losses is defined as follows in the file gaussian_diffusion.py

def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):

    # sample x_t from the unconditional forward process
    x_t = self.q_sample(x_start, t, noise=noise)

    # consider the MSE loss only 
    model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

    # get target from the reverse process
    target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
    

    terms["mse"] = mean_flat((target - model_output) ** 2)
    return terms

```

## How to train the classifier

In the following code snippet, we will go through the minimal code to train the classifier. The code is based on the file `classifier_train.py`. It is worth noting that the classifier can be trained on either training set or generated images from the diffusion model, controlled by the parameter `args.noised`

```python

def main():
    # create unet and scheduler of the diffusion model
    model, diffusion = create_model_and_diffusion()

    # init schedule sampler
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion)

    # create optimizer
    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0)

    # create unet model? repeat from previous step
    model = DDP(model, ...)

    # create data loader 
    data = load_data(...)

    # create optimizer
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)


```


References
=====

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR 2021.