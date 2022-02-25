# https://github.com/ChristophReich1996/Swin-Transformer-V2
from typing import Any

import torch
from PIL import Image
import os
import sys

from pipelines.super_resolution.magnifier import Magnifier

singleton: Magnifier


def load(config):
    global singleton
    print("loading...")
    if "model_name" in config and config["model_name"] not in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:
        print("unknown model name", config["model_name"])
    else:  # ['RealESRGAN_x4plus', 'RealESRNet_x4plus']
        from .bsrgan import BsrganMagnifier
        singleton = BsrganMagnifier()
    singleton.load(config)
    print("Finished loading!")


def transform(config, input_path, output_path):
    global singleton
    print("transforming...")
    singleton.transform(config, input_path, output_path)


def destroy(config):
    global singleton
    singleton = None
