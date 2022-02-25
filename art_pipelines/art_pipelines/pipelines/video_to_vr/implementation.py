import os

from .converter import Converter

singleton: Converter


def load(config):
    global singleton
    singleton = Converter()
    singleton.load(config)


def transform(config, input_path, output_path):
    global singleton
    output_stem, ext = os.path.splitext(output_path)
    if ext.lower() in [".mp4", ".gif"]:
        singleton.convert_video(input_path, output_path)  # , size=(640, 640))
    elif ext.lower() in [".jpg", ".png"]:
        singleton.convert_image(input_path, output_path)  # , size=(640, 640))


def destroy(config):
    global singleton
    singleton = None
