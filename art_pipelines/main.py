import importlib
import os
from typing import Dict

from tqdm.auto import tqdm

from art_pipelines.config_loader import config


def run_pipelines(config: Dict = config):
    for pipeline in config["pipelines"]:
        if "skip" in pipeline and pipeline["skip"]:
            continue
        pipeline_name = pipeline["name"]
        print("Loading pipeline impl ", pipeline_name)
        pipeline_impl = importlib.import_module(
            f"art_pipelines.pipelines.{pipeline_name}.implementation", package="."
        )
        pipeline_impl.load(config=pipeline)
        input_dir = os.path.abspath(pipeline["input_dir"])
        output_dir = os.path.abspath(pipeline["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"running {pipeline_name} on directory {input_dir}")
        # input_filepaths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]

        input_files = os.listdir(input_dir)
        if "include_types" in pipeline.keys():
            input_files = [
                input_file
                for input_file in input_files
                if os.path.splitext(input_file)[1].replace(".", "")
                in pipeline["include_types"]
            ]

        for input_file in tqdm(input_files):
            input_file_path = os.path.join(input_dir, input_file)
            print(f"running {pipeline_name} on file {input_file_path}")
            output_file_extension = (
                pipeline["output_ext"]
                if "output_ext" in pipeline.keys()
                else os.path.splitext(input_file_path)[1]
            )
            output_file_path = os.path.join(
                output_dir, f"{os.path.splitext(input_file)[0]}{output_file_extension}"
            )
            if (
                "overwrite" not in pipeline.keys() or not pipeline["overwrite"]
            ) and os.path.isfile(output_file_path):
                print(f"Skipping - exists: {output_file_path}")
                continue
            pipeline_impl.transform(
                config=pipeline,
                input_path=input_file_path,
                output_path=output_file_path,
            )
        pipeline_impl.destroy(config=config)
        print(pipeline)


if __name__ == "__main__":
    import argparse

    import yaml

    config_path = os.getenv("local_config_path", "./configs/dev.yaml")
    parser = argparse.ArgumentParser(
        description=f"Run the pipelines from {config_path}."
    )
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=config_path)

    args = parser.parse_args()
    print(f"loading config from {args.config_path}")
    assert os.path.isfile(
        args.config_path
    ), f"No config found at {os.path.abspath(args.config_path)}"
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if args.input_dir:
        config["pipelines"][0]["input_dir"] = args.input_dir
    print("Input directory:", config["pipelines"][0]["input_dir"])

    if args.output_dir:
        config["pipelines"][-1]["output_dir"] = args.output_dir
    print("Output directory:", config["pipelines"][-1]["output_dir"])

    print(config)
    run_pipelines(config)
