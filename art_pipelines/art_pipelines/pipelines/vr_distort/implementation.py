import os

import cv2
from tqdm.auto import tqdm

import image_utils
from .distortion_utils import distort_vr


def load(config):
    pass


def transform(config, input_path, output_path):
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
    with tqdm(total=total_frames) as pbar:
        while success and frame is not None:

            distorted_frame = distort_vr(
                image=frame,
                k1=config["k1"],
                k2=config["k2"],
                p1=config["p1"],
                p2=config["p2"],
                focal_length_x=config["focal_length_x"],
                focal_length_y=config["focal_length_y"],
            )
            distorted_frames.append(distorted_frame)

            success, frame = frame_generator.read()
            frames_processed += 1
            if frames_processed % 100 == 0:
                print("frames_processed", frames_processed)
            if frames_processed > total_frames:
                print("Too many frames!")
                break
            pbar.update(1)

    print("frames_processed", frames_processed)
    frame_generator.release()
    if output_path.endswith("mp4"):
        out = cv2.VideoWriter(
            output_path, fourcc, int(fps), (int(frame.shape[1]), int(frame.shape[0]))
        )
        print("writing video...")
        for frame in distorted_frames:
            out.write(frame)
        out.release()
    else:
        if frames_processed == 1:
            image_utils.save_image(
                image=distorted_frames[0],
                save_path=output_path
            )
        else:
            for i, frame in enumerate(distorted_frames):
                image_utils.save_image(
                    image=frame,
                    save_path=f"{os.path.splitext(output_path)[0]}_{i}{os.path.splitext(output_path)[1]}"
                )



def destroy(config):
    pass
