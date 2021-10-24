import os

import cv2

from art_pipelines.art_pipelines.pipelines.loop_video.loop_utils import (
    to_gif, to_mp4)


def load(config):
    pass


def transform(config, input_path: str, output_path: str):
    assert os.path.isfile(input_path)
    if output_path.endswith(".gif"):
        to_gif(input_path=input_path, output_path=output_path, **config)
    elif output_path.endswith(".mp4"):
        to_mp4(input_path=input_path, output_path=output_path, **config)
    else:
        raise NotImplementedError()


def destroy(config):
    pass
