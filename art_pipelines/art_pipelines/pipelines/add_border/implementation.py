import os
from typing import Tuple

import cv2


def load(config):
    pass


def transform(config, input_path: str, output_path: str):
    assert os.path.isfile(input_path)

    frame_generator = cv2.VideoCapture(input_path)
    success, frame = frame_generator.read()
    assert (
        success and frame is not None
    ), f"No frames read from {os.path.abspath(input_path)}"
    fps = frame_generator.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_frames = frame_generator.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total_frames", total_frames)
    frames_processed = 0
    distorted_frames = []
    while success and frame is not None:

        distorted_frame = add_border(
            image=frame,
            border_color=[255, 255, 255],
            border_width=1,
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


def add_border(image, border_color: Tuple[int, int, int], border_width: int = 1):
    border = cv2.copyMakeBorder(
        src=image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color,
    )
    return border
