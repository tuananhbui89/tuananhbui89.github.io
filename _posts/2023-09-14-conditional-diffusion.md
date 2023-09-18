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
    while inrange:
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

```

The Algorithm 2 (Conditional Sampling for DDIM, i.e., `ddim_sample_loop`) can be implemented as below. However, one point that I am still not understand is that the DDIM is the deterministic sampling method, however, in the code, at the end, we still sample from a Gaussian distribution with the mean as calculated in Algorithm 2. The sigma is controlled by additional parameter `eta`, where `eta=0` means truly deterministic sampling.

```python

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

## How to train the diffusion model

Training the diffusion model in this project is similar as in the DDPM or DDIM papers. Because even using auxiliary classifier, they are trained independently. The minimal code to train the diffusion model is as follows, which is based on the code from file [`image_train.py`](https://github.com/openai/guided-diffusion/blob/main/scripts/image_train.py) and [`train_util.py`](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/train_util.py#L22)


```python

```

## How to train the classifier

References
=====

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR 2021.