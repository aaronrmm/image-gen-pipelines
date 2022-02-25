import os
from typing import Union

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
                extension = image.format
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
