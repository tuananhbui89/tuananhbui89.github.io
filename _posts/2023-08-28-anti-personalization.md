---
title: 'On Reading: Anti-Personalization in Generative Models'
date: 2023-08-05
permalink: /posts/2023/08/papers/antipersonalization/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
  - Trustworthy machine learning
---
<br>


## Generative Watermarking Against Unauthorized Subject-Driven Image Synthesis

### About the paper

- Paper link: [https://arxiv.org/abs/2306.07754](https://arxiv.org/abs/2306.07754)

### Summary

- Problem setting: protection without sacrificing the utility of protected images for general (good) synthesis tasks. Unlike Anti-Dreambooth, where the goal is to completely prevent personalization, here the goal is to prevent unauthorized personalization with specific tasks (subject-driven synthesis).
- Real-world scenarios of misusing personalization generative models: 
  - **Copyright of Hollie Mengert**: a Reddit user published a DreamBooth model that is fine-tuned based on the artwork from an American artist Hollie Mengert. [link](https://www.reddit.com/r/StableDiffusion/comments/yaquby/2d_illustration_styles_are_scarce_on_stable/)
  - **Elon Musk is dating GM CEO Mary Barra**: [link](https://twitter.com/blovereviews/status/1639988583863042050)

- Threat Model:
  - What is a threat model in TML? Threat model is a description of the capabilities and goals of an adversary. It also can describe the environment in which the machine learning model and its adversary operate.
  - In this project, there are two parties involved: the subject owners and the subject synthesizers (benign or adversaries). The system includes a watermarking generator and a detector. There is also a public generative model (i.e., Stable Diffusion) that is used by the subject synthesizers.
  - The subject owners are the ones who want to protect their images from unauthorized personalization. In this project, the subject owners use the generative watermarking to generate watermarked images. Then they can track the potential unauthorized use by detecting if the watermark appears in synthesized images (then the watermark to be added and the watermark to be detected are different?). The subject owners have full access to the generator and the detector and can also further improve them by fine-tuning.
  - The subject synthesizers (benign or adversaries) are the ones who want to use the generative models to synthesize the target subject. The benign synthesizers are the ones who obtains authorization under the consent of the subject owners. The adversaries are the ones who want to synthesize the target subject without authorization. In this project, both benign and adversarial synthesizers have access to a public generative model (i.e., Stable Diffusion) and the protected/watermarked images.

![Avatar](../../images/AML/2306_07754_01.png)

Approach: 

Phase 1: Pre-training the watermarking generator and detector. Similar as GAN, there is an adversarial game between generator $G$ and detector $D$. The watermarked images denote as $x_w=x+w$. The goal of the generator is to generate watermarked images that are indistinguishable from the real images. Its objective loss:
$$L_G = \text{max}(LPIPS(x,x_w) - p, 0)$$
where *LPIPS* is the perceptual similarity metric [1], $p$ is a hyperparameter that controls the invisibility level, a smaller $p$ means a more invisible watermark.

The detector $D$ is trained to distinguish the watermarked images from the real images. Its objective loss:
$$L_D = -(1-y)\log(1-\hat{y}) - y \log(\hat{y})$$
where $y$ is the ground truth label, $\hat{y}$ is the predicted label.

Phase 2: Fine-tuning the detector with synthesized images. Using two set of images, $X$ and $X_w$, where $X_w$ is the watermarked images of $X$ to train corresponding personalized models $M$ and $M_w$ (i.e., with Textual Inversion or Dreambooth). Then using these models to generate synthesized images $S$ and $S_w$ with list of prompts. And use these images to fine-tune the detector $D$, where $S$ has label as real and $S_w$ has label as watermarked.

[1] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595. IEEE, 2018.