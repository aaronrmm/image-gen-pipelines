import os
from typing import List, Union

import numpy as np
import PIL.Image
import torch
from cv2 import cv2
from torchvision.utils import save_image as save_tensor_as_image


def save_image(
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        save_path=None,
        save_dir=None,
        save_name=None,
):
    if type(image) is PIL.Image.Image:
        if PIL.Image.isImageType(image):
            if not save_path:
                extension = image.format or "png"
                save_dir = os.path.abspath(save_dir)
                save_path = os.path.join(save_dir, f"{save_name}.{extension}")
            image.save(save_path)
        else:
            raise Exception(f"object to save must be an image, but is a {type(image)}")
    elif type(image) is torch.Tensor:
        #image: np.ndarray = image.cpu()
        print("shape 0", image.shape)
        #image = torch.sigmoid(image)  # output is the output tensor of your UNet, the sigmoid will center the range around 0.
        print("shape 1", image.shape)
        # Binarize the image
        #threshold = (image.min() + image.max()) * 0.5
        #image = torch.where(image > threshold, 0.9, 0.1)
        print("shape 3", image.shape)
        save_tensor_as_image(image, save_path)
        #cv2.imwrite(save_path, image)
        extension = "png"
        save_path = save_path or os.path.join(save_dir, f"{save_name}.{extension}")
    elif type(image) is np.ndarray:
        cv2.imwrite(save_path, image)
        extension = "png"
        save_path = save_path or os.path.join(save_dir, f"{save_name}.{extension}")
    else:
        raise NotImplementedError(f"Could not save image of type {type(image)}")
    if save_path:
        print(f"{save_name} saved to {save_path}")
        if os.path.isfile(save_path):
            print(f"Image successfully saved to {save_path}")
        else:
            print(f"Image failed to save to {save_path}")


def load_image(image_path):
    return PIL.Image.open(image_path)


def get_image_dimensions(image):
    if type(image) is PIL.Image.Image:
        pil_image: PIL.Image.Image = image
        return pil_image.width, pil_image.height
    else:
        raise NotImplementedError(
            f"{type(image)} is not implemented. Only PIL.Image.Image is implemented."
        )


def concatenate_images(images: List, axis="horizontal"):
    images = [
        PIL.Image.fromarray(
            np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=2),
            mode="RGB",
        )
        for image in images
    ]
    if axis == "horizontal":
        total_width = 0
        max_height = 0
        for image in images:
            width, height = get_image_dimensions(image)
            total_width += width
            max_height = max(max_height, height)
        canvas = PIL.Image.new("RGB", size=(total_width, max_height))
        next_x_coord = 0
        for image in images:
            width, _ = get_image_dimensions(image)
            canvas.paste(image, box=(next_x_coord, 0))
            next_x_coord += width
        return canvas
    else:
        raise NotImplementedError("Only horizontal image concatenation is implemented")
    return None
