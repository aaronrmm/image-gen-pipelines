import os

import PIL.Image


def save_image(image, save_dir, save_name):
    save_dir = os.path.abspath(save_dir)
    saved_path = None
    if PIL.Image.isImageType(image):
        extension = image.format
        save_path = os.path.join(save_dir, f"{save_name}.{extension}")
        image.save(save_path)
    else:
        raise Exception(f"object to save must be an image, but is a {type(image)}")
    if saved_path:
        print(f"{save_name} saved to {save_path}")


def load_image(image_path):
    return PIL.Image.open(image_path)
