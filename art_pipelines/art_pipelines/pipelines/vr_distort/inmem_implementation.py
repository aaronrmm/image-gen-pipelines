import os
from typing import Generator

import PIL.Image
import cv2
import numpy as np
from tqdm.auto import tqdm

from .distortion_utils import distort_vr


def load(config):
    pass


def transform(config, images):
    total_frames = len(images)
    print("total_frames", total_frames)
    distorted_frames = []
    for image in tqdm(images):
        distorted_frame = distort_vr(
            image=np.array(image),
            k1=config["k1"],
            k2=config["k2"],
            p1=config["p1"],
            p2=config["p2"],
            focal_length_x=config["focal_length_x"],
            focal_length_y=config["focal_length_y"],
        )
        distorted_frames.append(distorted_frame)

        if len(distorted_frames) > total_frames:
            print("Too many frames!")
            break

    print("frames_processed", len(distorted_frames))
    return distorted_frames


def destroy(config):
    pass
