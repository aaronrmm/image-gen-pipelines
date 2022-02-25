import math
import os
from typing import Tuple

import cv2
import numpy as np


def load(config):
    pass


def transform(config, input_path, output_path):
    assert os.path.isfile(input_path)
    aspect_ratio = config.get("ratio", "1:1")
    crop = config.get("crop", False)
    frame_generator = cv2.VideoCapture(input_path)
    success, frame = frame_generator.read()
    assert (
        success and frame is not None
    ), f"No frames read from {os.path.abspath(input_path)}"
    fps = frame_generator.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(
        output_path, fourcc, int(fps), (int(frame.shape[1]), int(frame.shape[0]))
    )

    total_frames = frame_generator.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total_frames", total_frames)
    frames_processed = 0
    distorted_frames = []
    while success and frame is not None:

        distorted_frame = (
            crop_image(image=frame, aspect_ratio=aspect_ratio)
            if crop
            else stretch(image=frame, aspect_ratio=aspect_ratio)
        )
        distorted_frames.append(distorted_frame)

        success, frame = frame_generator.read()
        frames_processed += 1
        if frames_processed % 100 == 0:
            print("frames_processed", frames_processed)
        if frames_processed > total_frames:
            print("Too many frames!")
            break

    print("frames_processed", frames_processed)
    frame_generator.release()

    output_stem, ext = os.path.splitext(output_path)
    if ext.lower() in [".mp4"]:
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            int(fps),
            (int(distorted_frames[0].shape[1]), int(distorted_frames[0].shape[0])),
        )
        print("writing video...")
        for frame in distorted_frames:
            out.write(frame)
        out.release()
    elif ext.lower() in [".png", ".jpg"]:
        cv2.imwrite(filename=output_path, img=distorted_frames[0])
        for n, image in distorted_frames[1:]:
            cv2.imwrite(
                filename=f"{output_stem}_{n}{ext}",
                img=distorted_frames[0],
            )


def destroy(config):
    pass


def stretch(image: np.ndarray, aspect_ratio: str = "1:1"):
    if aspect_ratio != "1:1":
        raise NotImplementedError()
    if aspect_ratio == "1:1":
        difference = image.shape[0] - image.shape[1]
        difference = max(difference, -difference)
        pads = difference // 2, math.ceil(difference / 2)
        top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0
        if image.shape[0] > image.shape[1]:
            right_pad, left_pad = pads
        else:
            top_pad, bottom_pad = pads
        return np.copy(
            np.pad(
                image,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )


def crop_image(image: np.ndarray, aspect_ratio: str = "1:1"):
    if aspect_ratio != "1:1":
        raise NotImplementedError()
    if aspect_ratio == "1:1":
        difference = image.shape[0] - image.shape[1]
        difference = max(difference, -difference)
        crop = difference // 2, math.ceil(difference / 2)
        new_top, new_bottom, new_left, new_right = 0, 0, 0, 0
        if image.shape[0] > image.shape[1]:
            new_image = np.copy(image[crop[0] : -crop[1]])
        else:
            new_image = np.copy(image[:, crop[0] : -crop[1]])
        return new_image
