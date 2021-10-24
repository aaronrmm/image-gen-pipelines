import argparse
import os

from .libs.photo_inpainting.boostmonodepth_utils import run_boostmonodepth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./images",
        help="path to directory containing images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./depths",
        help="path to directory for saving depth images",
    )
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    run_boostmonodepth(
        image_paths=[
            os.path.join(args.input_dir, image_filename)
            for image_filename in os.listdir(args.input_dir)
        ],
        depth_folder=output_dir,
    )
