import os
from typing import Any, List

import cv2
import PIL.Image

from ...image_utils import concatenate_images, save_image
from ..combine_video_frames.video_utils import get_video_frames, to_gif, to_mp4


def load(config):
    pass


def transform(config, input_path: str, output_path: str):
    assert os.path.isfile(input_path)
    if output_path.endswith(".gif") or output_path.endswith(".mp4"):
        frames: List[Any] = get_video_frames(input_path)
        if len(frames) < 1:
            print(f"Skipping video with no frames {input_path}")
            return
        image: PIL.Image.Image = concatenate_images(
            [frames[0], frames[-1]], axis="horizontal"
        )
        save_path = f"{os.path.splitext(output_path)[0]}.{'png'}"
        save_image(
            image,
            save_path=save_path
            # save_dir=output_path,
            # save_name=f"{os.path.basename(output_path)}.jpg",
        )
    else:
        raise NotImplementedError()


def destroy(config):
    pass
