---
layout: post
title: On Reading - Flow Matching for Generative Modeling
description: A cool paper about continuous normalizing flows
tags: reading genai
giscus_comments: true
date: 2023-09-22
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
#   - name: About the paper
#     # if a section has subsections, you can add them as follows:
#     # subsections:
#     #   - name: Example Child Subsection 1
#     #   - name: Example Child Subsection 2
#   - name: Approach
#   - name: How to implement
#   - name: Notes

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

## About the paper

- Published at ICLR 2023 (splotlight, top 5%)
- Affiliations: Meta AI, Weizmann Institute of Science
- Link to the paper: [https://openreview.net/pdf?id=PqvMRDCJT9t](https://openreview.net/pdf?id=PqvMRDCJT9t)

## Introduction

Continuous Normalizing Flows (CNF) is a class of generative models that can be trained by maximum likelihood. The main idea is to transform a simple distribution (e.g., Gaussian) to a complex distribution (e.g., ImageNet dataset) by a series of invertible transformations. The main challenge is to design a transformation that is invertible and can be computed efficiently.

The flow $\phi_t(x)$ presents a time-dependent diffeomorphic map that transforms the input $x$ to the output $y$ at time $t$. The flow is defined as follows:

$$ \frac{d}{dt} \phi_t(x) = v_t(\phi_t(x)) $$

where $v_t$ is a time-dependent vector field. $\phi_0(x) = x$ means that the flow at time $t=0$ is the identity map.

Given $p_0$ is the simple distribution (e.g., Gaussian), the flow $\phi_t$ transforms $p_0$ to $p_t$ as follows: 

$$ p_t = [ \phi_t ] * p_0 $$

where $[ \phi_t ] * p_0$ is the push-forward measure of $p_0$ under the map $\phi_t$. 
The push-forward measure is defined as follows:

$$ [ \phi_t ] * p_0(A) = p_0(\phi_t^{-1}(A)) $$

where $A$ is a subset of the output space. The push-forward measure can be interpreted as the probability of the output $y$ falls into the subset $A$.

$$ p_t(x) = p_0(\phi_t^{-1}(x)) \left| \det \frac{d \phi_t^{-1}(x)}{dx} \right| $$

The function $v_t$ can be intepreted as the velocity of the flow at time $t$, i.e., how fast the flow moves at time $t$. In comparison with diffusion process, the velocity $v_t$ is similar as the denoising function that is used to denoise the image $x$ at time $t$, where $\phi_t(x)$ is the distribution of the denoised images at time $t$.

<!-- The flow is invertible because it is a diffeomorphic map. The inverse flow is defined as follows:

$$ \frac{d}{dt} \phi_t^{-1}(y) = -v_t(\phi_t^{-1}(y)) $$
 -->

**Flow matching objective**: Given a target probability density path $p_t(x)$ and a corresponding vector field $u_t(x)$ which generates $p_t(x)$, the flow matching objective is to find a flow $\phi_t(x)$ and a corresponding vector field $v_t(x)$ that generates $p_t(x)$.

$$ \mathcal{L}_{FM} (\theta) = \mathbb{E}_{t, p_t(x)} \| v_t(x) - u_t(x) \| $$

It is a bit confusing in notation here, so $u_t(x)$ can be understand as the target vector field that generates the target probability density path $p_t(x)$, while $v_t(x)$ is the vector field to be learned to approximate $u_t(x)$.

It can be seen that the Flow Matching objective is a simple and attractive objective but intractable to use because we don't know the target vector field $u_t(x)$.
The main contribution of the paper is to propose the way to simplify the above objective function. And their approach is quite similar as in DDPM where the solution relies on conditioning to a previous point in the sequence.

The marginal probability path 

$$ p_t(x) = \int p_t(x \mid x_1) q(x_1) dx_1 $$ 

where $x_1$ is a particular data sample, and $p_t(x \mid x_1)$ is the conditional probability path such that $p_t(x \mid x_1) = p_t(x)$ at time $t=0$. 
The important point is that they design the $p_1(x \mid x_1)$ at time $t=1$ to be a normal distribution around $x_1$ with a small variance, i.e., $p_1 (x \mid x_1) = \mathcal{N}(x_1, \sigma^2 I)$. In the above equation, $q(x_1)$ is the prior distribution of $x_1$.

Where in particular at time $t=1$, the marginal probability path $p_!$ will approximate the data distribution $q$, 

$$ p_1(x) = \int p_1(x \mid x_1) q(x_1) dx_1 \approx q(x) $$ 

And the vector field $u_t(x)$ can be defined as follows: 

$$ u_t(x) = \int u_t(x \mid x_1) \frac{p_t (x \mid x_1) q(x_1)}{p_t(x)} dx_1 $$

Theorem 1: Given vector fields $u_t(x \mid x_t)$ that generate conditional probability paths $p_t(x \mid x_t)$ for any distribution $q(x_1)$, the marginal vector field $u_t(x)$ in the above equation generates the marginal probability path $p_t(x)$.

So it means that if we can learn $u_t (x \mid x_t)$ we can obtain $u_t(x)$ and then we can use $u_t(x)$ to generate $p_t(x)$.

Now we can rewrite the Flow Matching objective to Conditional Flow Matching objective as follows:

$$ \mathcal{L}_{CFM} (\theta) = \mathbb{E}_{t, q(x_1), p_t(x \mid x_1)} \| v_t(x) - u_t(x \mid x_1) \| $$

where $v_t(x)$ is the vector field to be learned to approximate $u_t(x \mid x_1)$. 
Now the question is how can we obtain $u_t(x \mid x_1)$?

In the work, they consider conditional probability paths 

$$ p_t(x \mid x_1) = \mathcal{N} (x \mid \mu_t (x_1), \sigma_t (x_1)^2 I) $$

where $\mu_t (x_1)$ and $\sigma_t (x_1)$ are the mean and variance of the conditional probability path $p_t(x \mid x_1)$, and they are time-dependent. Later, they will show that we can choose $\mu_t (x_1)$ and $\sigma_t (x_1)$ very flexiblely, as long as they can satisfy some conditions, for example, $\mu_0 (x_1) = 0$ and $\sigma_0 (x_1) = 1$, and $\mu_1 (x_1) = x_1$ and $\sigma_1 (x_1) = \sigma_{min}$, which is set sufficiently small so that $p_1 (x \mid x_1)$ is a concentrated distribution around $x_1$.

The canonical transformation for Gaussian distributions is defined as follows:

$$ \psi_t (x) = \mu_t (x_1) + \sigma_t (x_1) \odot x $$

where $\psi_t (x)$ is the canonical transformation of $p_t(x \mid x_1)$, and $\odot$ is the element-wise multiplication.

to be continued...