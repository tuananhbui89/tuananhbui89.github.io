---
layout: post
title: On Reading - Textual Inversion
description: Personalizing Text-to-Image Generation using Textual Inversion
tags: reading diffusion genai
giscus_comments: true
date: 2023-08-07
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
#   - name: How to implement
#     subsections:
#       - name: Specific Token
#       - name: Tokenizer thing
#       - name: Prompting process

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

- Published at ICLR 2023
- Affiliations: Tel Aviv University, Nvidia.
- Main idea: Every visual concept can be represented by a paragraph of text. The authors propose a method to learn a specific token that can represent a visual concept (It can learned so that with this specific token, the text-to-image can reconstruct the input images). The token is then used to generate a new image that contains the visual concept.  

## How to implement

In this blog post, I would like to break down some main steps in the implementation provided by Huggingface in the example code [here](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion). There are also two notebooks for training (learning conceptual token) and inference (using conceptual token to generate new images) [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) and [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb).

There are several main points as follows: 
- How to set up the specific token in the input prompt. 
- How to prepare the dataset 
- How to train and learn the specific token 

### How to set up the specific token and the input prompt

In this project, there is an assumption that every visual concept can be represented by a paragraph of text. For example, the visual concept of your dog can be described as: "The Shiba dog boasts a striking and distinctive appearance that captivates all who gaze upon it. With a compact yet sturdy build, its confident stance exudes an air of self-assured elegance. A plush double coat of fur, often seen in shades of red, sesame, black and tan, or cream, adds to its allure. The fur frames a fox-like face, adorned with piercing almond-shaped eyes that gleam with intelligence and curiosity. Its erect, triangular ears stand at attention, poised to catch every sound that graces its surroundings. A tightly curled tail rests gracefully over its back, accentuating the Shiba's poise and dignity." (I use ChatGPT to write this paragraph about a Shiba dog) 

However, if we use the entire paragraph to represent the visual concept, it is not efficient. Fortunately, we can just represent the whole paragraph by a single token (in the paper they used the token `S*`). 
So the first step is to set up a placeholder for the specific token in the input prompt. In the implementation, it can be set by the argument `placeholder_token` (default value is `<cat-toy>`).

There is also an argument `learnable_property` (option `object` or `style`) which is used to choose type of neural prompt from two sets of templates `imagenet_templates_small` (if set to `object`) or `imagenet_style_templates_small` (if set to `style`). 
Some examples of the templates are as follows:

```python
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
]
```

The `placeholder_token` will replace the `{}` in the templates. For example, if we set `placeholder_token` to `cat-toy` and `learnable_property` to `object`, the input prompt will be `a photo of a cat-toy`.
The input prompt then will be tokenized by the tokenizer.

```python
self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small

placeholder_string = self.placeholder_token

text = random.choice(self.templates).format(placeholder_string)

example["input_ids"] = self.tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=self.tokenizer.model_max_length,
    return_tensors="pt",
).input_ids[0]
```

### Tokenizer thing 

There is also an argument `initializer_token` (default value is `toy`) but not used anywhere else in the code. 

The `placeholder_token` (i.e., `<cat-toy>`) is converted to indexes by the tokenizer [`convert_tokens_to_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.convert_tokens_to_ids) method. Basically, this method converts the a sequence of tokens (i.e., `<cat-toy>`) in a sequence of ids, using the vocabulary.

```python
# Convert the initializer_token, placeholder_token to ids
token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

# Resize the token embeddings as we are adding new special tokens to the tokenizer
text_encoder.resize_token_embeddings(len(tokenizer))

# Initialise the newly added placeholder token with the embeddings of the initializer token
token_embeds = text_encoder.get_input_embeddings().weight.data
with torch.no_grad():
    for token_id in placeholder_token_ids:
        token_embeds[token_id] = token_embeds[initializer_token_id].clone()
```

The `placeholder_token_ids` is then used to specify the position in the embedding matrix to be updated (corresponding to our specific token). In the end, the only thing we need to learn is the embedding matrix (actually only several specific rows in the embedding matrix, but the entire embedding matrix is small enough to store/save unlike the unet' weights where it is much compacted using [LORA](https://huggingface.co/blog/lora)). Later, we will learn how to use the learned embedding matrix to generate new images or upload to the hub.

```python
# Let's make sure we don't update any embedding weights besides the newly added token
index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

with torch.no_grad():
    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
        index_no_updates
    ] = orig_embeds_params[index_no_updates]
```

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

### How to process the dataset

To be continued... :D

### How to train and learn the specific token

To be continued... :D

### How to package the learned token and use it for inference (or upload to the hub)

To be continued... :D