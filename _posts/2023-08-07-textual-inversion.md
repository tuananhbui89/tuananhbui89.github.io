---
title: 'On Reading: An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion'
date: 2023-08-05
permalink: /posts/2023/08/papers/textualinversion/
tags:
  - Generative model
  - Diffusion model
  - Paper reading
---
<br>

- [About the paper](#about-the-paper)
- [How to implement](#how-to-implement)
  - [How to set up the specific token and the input prompt](#how-to-set-up-the-specific-token-and-the-input-prompt)
    - [Tokenizer thing](#tokenizer-thing)
  - [How to process the dataset](#how-to-process-the-dataset)
  - [How to train and learn the specific token](#how-to-train-and-learn-the-specific-token)
  - [How to package the learned token and use it for inference (or upload to the hub)](#how-to-package-the-learned-token-and-use-it-for-inference-or-upload-to-the-hub)


About the paper
=====

- Published at ICLR 2023
- Affiliations: Tel Aviv University, Nvidia.
- Main idea: Every visual concept can be represented by a paragraph of text. The authors propose a method to learn a specific token that can represent a visual concept (It can learned so that with this specific token, the text-to-image can reconstruct the input images). The token is then used to generate a new image that contains the visual concept.  

How to implement
=====

In this blog post, I would like to break down some main steps in the implementation provided by Huggingface in the example code [here](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion). There are also two notebooks for training (learning conceptual token) and inference (using conceptual token to generate new images) [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) and [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb).

There are several main points as follows: 
- How to set up the specific token in the input prompt. 
- How to prepare the dataset 
- How to train and learn the specific token 

## How to set up the specific token and the input prompt

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

## How to process the dataset 

The above step shows how the specific token is processed. In this step, we will see in a higher level of how the training dataset is processed.


## How to train and learn the specific token 


## How to package the learned token and use it for inference (or upload to the hub)