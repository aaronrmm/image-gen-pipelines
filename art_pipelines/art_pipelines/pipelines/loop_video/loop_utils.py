import os

import cv2
from PIL import Image


def to_mp4(
    input_path: str, output_path: str, min_duration: float = None, num_loops: int = None
):
    assert os.path.isfile(input_path)
    assert (
        min_duration or num_loops
    ), "Must specify a target duration or number of loops"

    frame_generator = cv2.VideoCapture(input_path)
    success, frame = frame_generator.read()
    assert (
        success and frame is not None
    ), f"No frames read from {os.path.abspath(input_path)}"

    fps = frame_generator.get(cv2.CAP_PROP_FPS)
    total_frames = frame_generator.get(cv2.CAP_PROP_FRAME_COUNT)

    num_loops = (
        num_loops
        if num_loops > -1
        else get_loops_required(duration=min_duration, fps=fps, num_frames=total_frames)
    )
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        int(fps),
        (int(frame.shape[1]), int(frame.shape[0])),
    )
    frames_processed = 0
    frames = []
    while success and frame is not None:
        frames.append(frame)

        success, frame = frame_generator.read()
        frames_processed += 1
        if frames_processed > total_frames:
            print("Too many frames!")
            break
    frame_generator.release()
    frames *= num_loops
    print("writing video...")
    for frame in frames:
        out.write(frame)
    out.release()


def to_gif(
    input_path: str, output_path: str, min_duration: float, num_loops: int = -1
) -> object:
    video_capture = cv2.VideoCapture(input_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    still_reading, image = video_capture.read()
    frame_count = 0
    frames = []
    while still_reading:
        frames.append(image)
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1
    video_capture.release()
    assert len(frames) == total_frames
    if len(frames) < 2:
        print("skipping")
        return

    num_loops = (
        num_loops
        if num_loops > -1
        else get_loops_required(duration=min_duration, fps=fps, num_frames=total_frames)
    )
    pil_images = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = Image.fromarray(image)
        pil_images.append(new_frame)
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 / fps,
        loop=int(num_loops),
    )


def get_loops_required(duration: float, num_frames: int, fps: int = 30):
    loops_per_duration = max(1, int(duration * fps / num_frames))
    return loops_per_duration
