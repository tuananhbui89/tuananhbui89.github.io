---
layout: post
title: Some useful code snippets
description: Not something fancy
tags: coding
giscus_comments: true
date: 2023-08-01
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


To log some useful code snippets that I have learned and used.

## Write an image from array

Need to convert to uint8 before writing to image to avoid color shift.
And using `cv2` will cause color shift (I didn't try my best to find out why, there might be some other ways to use `cv2` to write image without color shift).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/code/write_image_PIL.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/code/write_image_cv2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Writing image using PIL and cv2.
</div>


```python
    from PIL import Image
    import torch 

    def save_images(images, save_path):
        assert(torch.is_tensor(images))
        assert(len(images.shape) == 4)
        assert(images.shape[1] == 3)
        assert(torch.min(images) == -1)
        assert(torch.max(images) == 1)

        for id, img_pixel in enumerate(images):
            save_path_id = save_path + str(id) + ".png"
            Image.fromarray(
                (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path_id)

            # Using cv2 will cause color shift
            # cv2.imwrite(save_path_id, (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
    
    if __name__ == "__main__":
        images = torch.load("https://raw.githubusercontent.com/tuananhbui89/tuananhbui89.github.io/master/files/images_tensor.pt")
        save_path = "./test/"
        save_images(images, save_path)
```