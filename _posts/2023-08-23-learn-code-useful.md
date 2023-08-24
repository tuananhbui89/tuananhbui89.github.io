---
title: 'Useful code snippets'
date: 2023-08-23
permalink: /blog/usefulcode/
tags:
  - Learning
---
<br>

To log some useful code snippets that I have learned and used.

## Write an image from array

Need to convert to uint8 before writing to image to avoid color shift.
And using `cv2` will cause color shift (I didn't try my best to find out why, there might be some other ways to use `cv2` to write image without color shift).

![PIL](../../images/code/write_image_PIL.png) Write image using PIL.

![cv2](../../images/code/write_image_cv2.png) Write image using cv2.


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