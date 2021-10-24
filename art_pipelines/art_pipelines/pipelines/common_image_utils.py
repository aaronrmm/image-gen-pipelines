import cv2
import PIL
import numpy as np
from PIL.Image import Image as Pil_Image


def load_pil_image(input_path) -> Pil_Image:
    image = PIL.Image.open(input_path)
    return image


def save_pil_image(image: Pil_Image, output_path):
    image.save(output_path)


def load_cv2_image(input_path)->np.ndarray:
    img: np.ndarray = cv2.imread(input_path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img
