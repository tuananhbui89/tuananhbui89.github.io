---
title: 'Learning and Thinking Log'
date: 2023-07-21
permalink: /posts/2023/07/learning/learninglog/
tags:
  - Learning
---
<br>

Learning targets 
--------------------------
This is a learning log to keep track of what I have learned each day. To avoid distraction and fluctuation, I will set some long-term learning targets that needed a disciplined and consistent effort to achieve.

- #Productivity: learn how to be more productive and effective in work and life. (effective != efficient, outcome != output, proactive > active > reactive, 1.01^365 = 37.8, 0.99^365 = 0.03)
- #AML: learn more about adversarial machine learning.
- #GenAI: learn more about generative AI, more specifically, personalized AI.
- #F4T: food for thought, learn more about philosophy, history, and humanity.
- #Finance: learn more about finance, investment, and business.

Short-term targets (Updated on July 2023):

- #Coding: Textual Inversion and Dreambooth
- #Research: Writing a paper about TML in GenAI

<!-- 
Recording ideas
--------------------------
Starting on 08/08/2023, I have decided to log all of my **dumb** ideas and thoughts here. In the past three years, since I began my Ph.D., I have recorded many ideas in my OneNote but have not shared them with anyone. It was because I thought that: A- they were just too dumb to share and B- (very few) they were just too good to share :D. However, after revisting old notes, I have come to realize that A- some of them were not that dumb, and even dumb so who cares, and B- some of them were not that good, and after a while, I still did not do anything with them. Therefore, instead of letting these ideas die in my notebook, I believe it is better to expose them to the light of the Internet. I hope this experiment will be long-lasting, fun, and useful.

Disclaimer:

- Originality: I would also like to emphasize that most of my research ideas come to mind when I am reading papers or watching lectures, and I do not always have the time to do a thorough literature review. Therefore, it is possible that some of my ideas have already been proposed by someone else without my knowledge.
- Stupidity: when I say dumb ideas, I mean that they are not well thought out, and even so they came from a not-so-smart with limited knowledge person. Therefore, I would also like to emphasize that I am not responsible for any damage caused by them :D. -->

2023-08-16
--------------------------
(#Research) The Inaproppriate Image Prompts (I2P) benchmark.

- Including 4703 unique prompts to generate inappropriate images with Stable Diffusion. There are combinations of 7 categories including: hate, harrassment, violence, self-harm, sexual, shocking and illegal activities.
- Research paper: [Safe Latent Diffusion:
Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105), CVPR 2023.
- Huggingface page: [https://huggingface.co/datasets/AIML-TUDA/i2p](https://huggingface.co/datasets/AIML-TUDA/i2p)

2023-08-14
--------------------------
(#Research) Some trends in KDD 2023: Graph Neural Networks and Casual Inference from Industrial Applications.

(#Research) Graph Neural Networks, definition of neighborhood aggregation. Most of GNN methods work on million of nodes, to scale to billion of nodes, there are a lot of tricks under the hood (from Dinh's working experience in Trustingsocial).

(#Research) (With Trung and Van Anh) We derive a nice framework that connect data-space distributional robustness (as in our ICLR 2022 paper) and model-space distributional robustness (as in SAM).

2023-08-08
--------------------------
(#Research) On reading: Erasing Concepts from Diffusion Models (ICCV 2023).  [https://erasing.baulab.info/](https://erasing.baulab.info/)

(#Research) On reading: CIRCUMVENTING CONCEPT ERASURE METHODS FOR TEXT-TO-IMAGE GENERATIVE MODELS. Project page: [https://nyu-dice-lab.github.io/CCE/](https://nyu-dice-lab.github.io/CCE/) 

2023-08-06
--------------------------
(#Coding) Strange bug in generating adversarial examples using Huggingface.

- Context: I am trying to implement similar idea as in the [Anti-Dreambooth project](https://anti-dreambooth.github.io/) to generate adversarial perturbation for the Textual Inversion project. However, I got this bug which cost me 2 days to figure out. 
- Bug: the gradient on the input tensor is None, even required_grad is set to True.
- Cause: Because of the Huggingface accelerator. (Ref: [gradient_accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)). The accelerator is to help to accelerate the training process by accumulating the gradient over multiple batches. It requires the model and the train dataloader to be prepared with function `accelerator.prepare()`. However, in our case, we do not use the train dataloader (see the [blog post about Anti-Dreambooth](https://tuananhbui89.github.io/posts/2023/08/papers/antidreambooth/)). Therefore, when we still use the accelerator.accumulate() in the training loop, the gradient is accumulated over multiple batches, and the gradient on the input tensor is None.
- Fix: remove the accelerator.accumulate() in the training loop. No it doesn't work! 

2023-08-05
--------------------------
(#Coding) Understand the implementation of the [Anti-Dreambooth project](https://anti-dreambooth.github.io/). Ref to the [blog post](https://tuananhbui89.github.io/posts/2023/08/papers/antidreambooth/)


2023-08-04 
--------------------------
(#Research) Three views of Diffusion Models:

- Probabilistic view point as in DDPM
- Denoising Score matching
- Stochastic Differential Equations (SDE) view point (which is the most general one)

2023-08-03
--------------------------
(#Research) Trusted Autonomous Systems

- Trusted Autonomous Systems (TAS) is Australia’s first Defence Cooperative Research Centre.
- There are many trustworthy related projects undergoing in this center.  
- Reference: [https://tasdcrc.com.au/ ](https://tasdcrc.com.au/about-us/)

2023-08-01
--------------------------
(#Research) Helmholtz Visiting Researcher Grant

- https://www.helmholtz-hida.de/en/new-horizons/hida-visiting-program/ 
- 1-3 months visiting grant for Ph.D. students and postdocs in one of 18 Helmholtz centers in Germany. 
- Deadline: 16 August 2023 and will end on 15 October 2023.
- CISPA - Helmholtz Center for Information Security https://cispa.de/en/people 

2023-07-31
--------------------------
(#Research) Australia Research Council (ARC) Discovery Project (DP) 2023.

- The ARC DP is a very competitive grant in Australia. 
- List of successful funded projects in 2023: https://rms.arc.gov.au/RMS/Report/Download/Report/1b0c8b2e-7bb0-4f2d-8f52-ad207cfbb41d/243 
- This is a good source to find potential postdoc positions or Ph.D. scholarships as well as to find potential collaborators. 
- For example, regarding Trustworthy Machine Learning, there are two projects including Dinh's project. 



2023-07-30
--------------------------
() First Home Buyer Super Saver Scheme.

- Save mony for first home buyer inside superannuation fund. Apply 15% tax rate instead of marginal tax rate. When withdraw, apply another 15% tax rate. Each individual can save up to 50k for cross all years. 


2023-07-27
--------------------------

(#GenAI) How to run textual inversion using Huggingface library locally without login to Huggingface with token. Including:

- How to install git-lfs on Linux to download large files from Github. Reference: https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md 
- How to download pretrained model from Huggingface model hub. Reference: https://huggingface.co/docs/hub/models-downloading 
- Setup environment to run textual inversion locally. Setup dependencies. 
- Setup script

2023-07-24
--------------------------

(#Productivity) How to present a slide and take notes on the same screen simultaneously (e.g., it is very useful when teaching or giving a talk). At Monash, the lecture theatre has MirrorOp installed on all screens that can connect wirelessly with a laptop but it is not convenient when we want to take notes.

- Best solution: connect Ipad to the screen and use Ipad to present the slide. We can also see the slide's note on the Ipad (required an adapter USB-C to HDMI and a HDMI cable).
- Alternative solution: join Zoom meeting on personal computer (for presentation) and on Ipad (for taking notes) and share screen from Ipad on Zoom if needed. PC can connect to the screen using HDMI cable or MirrorOp. 


2023-07-23
--------------------------

Micromouse competition.

- First introduced by Claude Shannon in 1950s. 
- At the begining, it was just a simple maze solving competition. However, after 50 years of growing and competing, it has become a very competitive competition with many different categories: speed, efficiency, size. And along with its, many great ideas have been introduced and applied to the competition. It involes many different fields: mechanical, electrical, software, and AI all in just a small robot. 
- The Fosbury Flop in high jump. When everyone use the same jump technique, the performance becomes saturated. Then Fosbury introduced a new technique (backward flop) that no one had ever thought of before. And it became the new standard (even named after him). This phenomenon also happens in the Micromouse competition.
- The two most important game changing ideas in the history of micromouse competition: capability to diagonal movement and using fan (vacumn) to suck the mouse to the path so that the mouse can move faster as in a racing car.
 
Reference:

- [The Fastest Maze-Solving Competition On Earth by Veritasium.](https://youtu.be/ZMQbHMgK2rw)  
- [The Fosbury Flop—A Game-Changing Technique](https://invention.si.edu/fosbury-flop-game-changing-technique)

2023-07-22
--------------------------

(#Parenting) ATAR and university admission. 

(#AML) Rethinking Backdoor Attacks, Mardy's group. 
https://arxiv.org/pdf/2307.10163.pdf

(#Productivity) How to synchonize PowerPoint in Teams among multiple editers working simultaneously (i.e., share file in Teams, and open file in local using PowerPoint), in this way can retain the math equations. 

If you have some math equations in your PowerPoint, and open it on Google Slide, the equations will be converted to images. If you accidently sync the file with your original file, all math equations will be lost.

2023-07-21
--------------------------

(#F4T) You cannot solve a problem with the same thinking that created it. Albert Einstein. 

Context: Our lab had a workshop last week and Dinh gave a talk about his favorite book "The 7 habits of highly effective people". One of the habits is "Sharpen the saw" means that you always need to improve yourself from all aspects: physical, mental, spiritual, and social. That is the way you can overcome your limits and obstacles that you are facing.  

- You cannot solve a research problem in your thesis with the same knowledge that you have when starting your thesis. You need to grow and learn. 
- If you want to upgrade for your paycheck, you need to learn new skills, new knowledge before applying for a new job. 

(#Experience) The first department metting as a new Research Fellow. 

Context: I have just started my new position as a RF at the Department of Data Science and AI, Monash University. Today is the first time to be exposed to what really happen beyond student's perspective.

- Teaching matter (because new semester is about to start)
- Head of departmemt's presentation about current activities (especially hiring and open positions, and budget)
- Ph.D. student recruitment, how competitive it is and how to rank the candidates (some insights: academic record and research record)

