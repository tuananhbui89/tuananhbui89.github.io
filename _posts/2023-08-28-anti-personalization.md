---
layout: post
title: On Reading - Anti-Personalization in Generative Models
description: But just two papers so far :(
tags: reading diffusion genai tml
giscus_comments: true
date: 2023-08-23
featured: false

# authors:
#   - name: Tuan-Anh Bui
#     url: "https://tuananhbui89.github.io/"
#     affiliations:
#       name: Monash University

# bibliography: 2023-06-02-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
# toc:
#   - name: from Charlie Munger
#   - name: from Naval Ravikant
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }

toc:
  beginning: true
---

## Anti-Dreambooth

Link to other post: [https://tuananhbui89.github.io/blog/2023/anti-dreambooth/](https://tuananhbui89.github.io/blog/2023/anti-dreambooth/)

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


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/AML/2306_07754_01.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The Framework
</div>

Approach: 

Phase 1: Pre-training the watermarking generator and detector. Similar as GAN, there is an adversarial game between generator $G$ and detector $D$. The watermarked images denote as $x_w=x+w$. The goal of the generator is to generate watermarked images that are indistinguishable from the real images. Its objective loss:
$$L_G = \text{max}(LPIPS(x,x_w) - p, 0)$$
where *LPIPS* is the perceptual similarity metric [1], $p$ is a hyperparameter that controls the invisibility level, a smaller $p$ means a more invisible watermark.

The detector $D$ is trained to distinguish the watermarked images from the real images. Its objective loss:
$$L_D = -(1-y)\log(1-\hat{y}) - y \log(\hat{y})$$
where $y$ is the ground truth label, $\hat{y}$ is the predicted label.

Phase 2: Fine-tuning the detector with synthesized images. Using two set of images, $X$ and $X_w$, where $X_w$ is the watermarked images of $X$ to train corresponding personalized models $M$ and $M_w$ (i.e., with Textual Inversion or Dreambooth). Then using these models to generate synthesized images $S$ and $S_w$ with list of prompts. And use these images to fine-tune the detector $D$, where $S$ has label as real and $S_w$ has label as watermarked.

[1] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595. IEEE, 2018.