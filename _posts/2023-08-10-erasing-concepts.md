---
title: 'On Reading: Erasing Concepts from Diffusion Models'
date: 2023-08-05
permalink: /posts/2023/08/papers/erasingconcepts/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
---
<br>

- [About the paper](#about-the-paper)
- [How to implement](#how-to-implement)

About the paper
=====

- Published at ICCV 2023
- Affiliations: Northeastern University and MIT.
- Motivation: Remove specific concepts from diffusion models weights. The concept can be a specific style (i.e., nudity, Van Gogh style, etc.) or a specific object (i.e., car, dog, etc.) while preserving the rest of the output.
- Main idea:
<!-- Add an image -->
|![Examples](https://raw.githubusercontent.com/tuananhbui89/website_images/master/posts/erasing_concepts/examples.png)|
|:--:|
| *Examples of erasing nudity, Van Gogh style or an objects from a Stable Diffusion model (Image source: [Gandikota et al. (2023)](https://erasing.baulab.info/))* |
- Project page: [https://erasing.baulab.info/](https://erasing.baulab.info/)



How to implement
=====
