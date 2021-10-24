import os
from typing import Union

import numpy as np
import PIL.Image
from cv2 import cv2


def save_image(
    image: Union[PIL.Image.Image, np.ndarray],
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
    elif type(image) is np.ndarray:
        cv2.imwrite(save_path, image)
        extension = "png"
        save_path = save_path or os.path.join(save_dir, f"{save_name}.{extension}")
    if save_path:
        print(f"{save_name} saved to {save_path}")


def load_image(image_path):
    return PIL.Image.open(image_path)
