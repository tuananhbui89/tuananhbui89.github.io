---
title: 'An Overview of Adversarial Machine Learning - Part 1'
date: 2023-06-07
permalink: /posts/2023/06/overview/
tags:
  - AML
  - tutorial
---

# Contents 

- [Contents](#contents)
- [Trustworthy Machine Learning vs Adversarial Machine Learning](#trustworthy-machine-learning-vs-adversarial-machine-learning)
- [Adversarial Attacks - Manipulating ML Models](#adversarial-attacks---manipulating-ml-models)
  - [Poisoning Attacks](#poisoning-attacks)
  - [Backdoor Attacks](#backdoor-attacks)
  - [Model Extraction](#model-extraction)
  - [Privacy Attacks](#privacy-attacks)
  - [Evasion Attacks (Adversarial Examples)](#evasion-attacks-adversarial-examples)
  - [Some thought on the practicality of adversarial attacks](#some-thought-on-the-practicality-of-adversarial-attacks)
- [Adversarial Defenses - Protecting ML Models](#adversarial-defenses---protecting-ml-models)
- [Certified Robustness - Formal Guarantees Against Adversarial Attacks](#certified-robustness---formal-guarantees-against-adversarial-attacks)
- [AML for Good - Practical Applications and Positive Impacts](#aml-for-good---practical-applications-and-positive-impacts)
- [References](#references)

Reference: This post is inspired by the tutorial "Adversarial Machine Learning for Good" by Dr. [Pin-Yu Chen](https://sites.google.com/site/pinyuchenpage/home) (IBM Research) at the AAAI 2022 conference. Link to the tutorial: https://sites.google.com/view/advml4good

Trustworthy Machine Learning vs Adversarial Machine Learning
======

Trustworthy Machine Learning (TML) is a broad area of research that focuses on developing and deploying reliable, transparent, fair and secure machine learning systems. While Adversarial Machine Learning (AML) is a subfield of TML which specifically focuses on defending against malicious attacks. Let's compare these two concepts in more detail:

Trustworthy Machine Learning:

- Objective: TML aims to ensure the reliability, transparency, fairness, and security of machine learning models and algorithms.
- Scope: TML encompasses a broader range of considerations, including data quality, model interpretability, algorithmic bias, privacy, and fairness in decision-making.
- Research Focus: TML researchers investigate techniques and methodologies to enhance the overall trustworthiness of machine learning systems, taking into account ethical, legal, and societal implications.
- Application Areas: TML is relevant across various sectors, such as healthcare, finance, autonomous vehicles, and cybersecurity, where the consequences of unreliable or biased machine learning predictions can have significant impacts.

Adversarial Machine Learning:

- Objective: AML focuses on understanding and defending against adversarial attacks, where malicious actors manipulate input data to deceive or exploit vulnerabilities in machine learning models.
- Scope: AML primarily concerns itself with studying and countering specific threats to machine learning systems, including evasion attacks, poisoning attacks, model extraction, and membership inference attacks.
- Research Focus: AML researchers explore techniques to detect, mitigate, and recover from adversarial attacks, aiming to enhance the resilience and robustness of machine learning models.
- Application Areas: AML is particularly relevant in domains where the impact of successful attacks can be severe, such as autonomous vehicles, cybersecurity systems, fraud detection, and sensitive data analysis.

In this post, we will focus on AML research and discuss four main topics: adversarial attacks, adversarial defenses, certified robustness, and AML for good.

Adversarial Attacks - Manipulating ML Models
======

Adversarial attacks aim to manipulate machine learning models by exploiting their vulnerabilities. 
They can be categorized based on the following criteria:

- **Attack Objective**: What is the goal of the attack?
- **Attack Knowledge**: What is the attacker's knowledge about the target model? There are two types of settings: white-box and black-box. In the white-box setting, the attacker knows everything of the target model, including its architecture, parameters, data, and training process. It even knows defending strategies and can adapt its attack accordingly. In the black-box setting, the attacker has no access to the model's internal information. It can only query the model and observe its outputs (probabilities or just final predictions).
- **Attack Phase**: What is the attacker's access to the target model? There are three types of access: data, training, and inference. In the data access setting, the attacker can manipulate the input data before feeding it to the target model. In the training access setting, the attacker can manipulate the training data or the training process of the target model. In this case, the aim is to corrupt/manipulate the target model's parameters (i.e., poisoning/backdoor attacks). In the inference access setting, the target model is fixed, and the attacker can only manipulate the input data. 


|![Attack Category](https://raw.githubusercontent.com/tuananhbui89/website_images/master/posts/aml_intro/attack_category_3.png)|
|:--:|
| *Category of Adversarial attacks based on their access to the data, training or inference process of a target model. Adapted from Pin-Yu Chen's tutorial (2022)* |

Based on the above criteria, there are many types of adversarial attacks. In this post, we will focus on the most common ones: poisoning, backdoor attacks, model extraction, evasion attacks, and privacy attack. 

## Poisoning Attacks 

Aim: Poisoning attacks aim to manipulate the training data or the training process of a target model to corrupt its parameters. The attacker's goal is to make the target model misbehave on future test data. 


Threat scenario: StabilityAI and OpenAI are two competing companies that develop image generation models. StablityAI has a better model than OpenAI. However, OpenAI wants to win the competition. Therefore, it hires a group of hackers to hack into StabilityAI's training process and corrupt its model. As a result, StabilityAI's model is corrupted, and it starts to generate images that are not as good as before. OpenAI wins the competition.

Some notable works: [1, 2, 3]

## Backdoor Attacks 

Aim: Backdoor attacks aim to manipulate the training data or the training process of a target model to embed a backdoor into the model. The attacker's goal is to make the target model misbehave on future test data that contains a trigger, where the trigger is a specific pattern that is not present in the training data (recently, there are some works that can embed a backdoor with imperceptible triggers). Without the trigger, the model behaves normally. However, when the trigger is present in the input data, the model starts to make wrong predictions. In order to make the attack's threat more practical, the attacker is allowed to control the training data only, not the training process. However, it is still possible to embed a backdoor into the model by manipulating the training process or specific architecture.

Threat scenario: A bank wants to use a machine learning model to predict whether a loan applicant will default or not. The bank hires a data scientist to develop a machine learning model for this task. However, the data scientist is not honest. He/she wants to make the model misbehave on his/her future loan application. Therefore, he/she embeds a backdoor into the model. As a result, the model starts to make wrong predictions on future test data that contains a trigger. The data scientist gets benefits from this attack.

Some notable works: [4, 5, 6]

## Model Extraction 

Aim: Model extraction attacks aim to extract a copy of a target model. The attacker's goal is to obtain a model that has similar performance to the target model.

Threat scenario: A bank has a machine learning model that can predict whether a loan applicant will be accepted or not. The bank wants to keep this model secret. However, a competitor wants to obtain this model. Therefore, the competitor hires a hacker that can submit a lot of queries and replicate the model from observed output. They can use this model for their own benefits. 


Some notable works: [7, 8, 9]


## Privacy Attacks

Aim: Privacy attacks aim to extract sensitive information from a target model. The attacker's goal is to obtain sensitive information from the target model.

Threat scenario: A bank trained their chatbox using their clients' data and release their chatbox for publish use. However, a competitor wants to obtain their clients' data. Therefore, the competitor hires a hacker that interact with the chatbox and extract the clients' data. They can use this data for their own benefits.

Some notable works: Recent work from Carlini et al. [10] demonstrates that it is possible to extract training data from Large Language Models (LLMs) such as GPT-2 and GPT-3. Another work from [11] shows that generative model trained by copyrighted/unallowed images. 

## Evasion Attacks (Adversarial Examples)

Aim: Evasion attacks aim to manipulate the input data to cause a target model to make a incorrect prediction. The common goal is to make the model predict a specific class (targeted attack) or make the model predict any class other than the correct class (untargeted attack). The perturbation is usually small and imperceptible to human perception. 

Threat scenario: A eKYC system uses a machine learning model to verify the identity of a person. However, a hacker wants to bypass this system. Therefore, he/she manipulates his/her ID card to make the system misbehave. As a result, the system accepts his/her ID card, and he/she can get access to the system. It is even worse if the system is used in warfare. 

Some notable works: Szegedy et al. [12] first demonstrated the existence of adversarial examples. Madry et al. [13] proposed a PGD attack. 


## Some thought on the practicality of adversarial attacks

While adversarial attacks are definitely a big issue when it comes to deploying machine learning models, in my opinion, the situation isn't as bad as it seems. That's probably why many companies, even though they know about these attacks, don't really take proper actions to defend against them. (Except for a few like Google, Microsoft, and Facebook, but they're not the majority. I'm exhausted from searching for job opportunities in the industry, and let me tell you, most companies just don't care about adversarial attacks.)

When it comes to poisoning attacks and backdoor attacks, attackers need access to the training data or the training process, which isn't always possible. They can't control how a model is trained. In model extraction attacks, attackers have to submit a bunch of queries to the target model, which isn't always doable. And let's talk about privacy attacks - current research successfully extracts some training data, but guess what? That data is already out there on the internet, so it's not really a big deal. Adversarial examples, on the other hand, are a real headache when deploying machine learning models. But white-box attacks? They're not practical because the attacker needs to know everything about the target model. Black-box attacks are more realistic, but even then, they either require a ton of queries or rely on transferability.

So, while adversarial attacks are a genuine concern, I think it's important to see the bigger picture. Many companies don't prioritize defending against these attacks. Certain limitations exist for different types of attacks, which can make them less feasible or impactful. Adversarial examples remain a significant challenge, but let's not forget that ongoing research is working on developing stronger defenses. As the industry grapples with these attacks, I hope we'll see more secure and reliable machine learning models in the future.


Adversarial Defenses - Protecting ML Models
======

We will discuss in the next post (Part 2).

Certified Robustness - Formal Guarantees Against Adversarial Attacks
======

We will discuss in the next post (Part 2).

AML for Good - Practical Applications and Positive Impacts 
======

We will discuss in the next post (Part 2).

References
======

[1] Battista Biggio, Blaine Nelson, and Pavel Laskov. Poisoning attacks against support vector machines. ICML, 2012. 

[2] Ali Shafahi, W Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, and Tom Goldstein. Poison frogs! targeted clean-label poisoning attacks on neural networks. NeurIPS, 2018. 

[3] Jacob Steinhardt, Pang Wei W Koh, and Percy S Liang. Certiﬁed defenses for data poisoning attacks. NeurIPS, 2017. 

[4] Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Identifying vulnerabilities in the machine learning model supply chain. arXiv preprint arXiv:1708.06733, 2017. 

[5] Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai, Weihang Wang, and Xiangyu Zhang. Trojaning attack on neural networks. NDSS, 2018. 

[6] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526, 2017. 

[7] Jean-Baptiste Truong, Pratyush Maini, Robert J Walls, and Nicolas Papernot. Data free model extraction. CVPR, 2021. 

[8] Hengrui Jia, Christopher A Choquette-Choo, Varun Chandrasekaran, and Nicolas Papernot. Entangled watermarks as a defense against model extraction. USENIX, 2021. 

[9] Osbert Bastani, Carolyn Kim, and Hamsa Bastani. Interpreting blackbox models via model extraction. arXiv preprint arXiv:1705.08504, 2017a.

[10] Carlini et al. Extracting Training Data from Large Language Models, USENIX, 2021. 

[11] Carlini, Nicholas, et al. "Extracting training data from diffusion models." arXiv preprint arXiv:2301.13188 (2023).

[12] Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199.

[13] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).