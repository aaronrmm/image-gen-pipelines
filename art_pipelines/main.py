import importlib
from typing import Dict
from art_pipelines.config_loader import config
import os
from tqdm.auto import tqdm


def run_pipelines(config: Dict = config):
    for pipeline in config["pipelines"]:
        pipeline_name = pipeline["name"]
        pipeline_impl = importlib.import_module(
            f"art_pipelines.pipelines.{pipeline_name}.implementation",
            package="."
        )
        pipeline_impl.load(config=config)
        input_dir = os.path.abspath(pipeline['input_dir'])
        output_dir = os.path.abspath(pipeline['output_dir'])
        print(f"running {pipeline_name} on directory {input_dir}")
        #input_filepaths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]

        input_files = os.listdir(input_dir)
        if "include_types" in pipeline.keys():
            input_files = [
                input_file for input_file in input_files
                if os.path.splitext(input_file)[1].replace(".","") in pipeline["include_types"]
            ]

        for input_file in tqdm(input_files):
            input_file_path = os.path.join(
                input_dir,
                input_file
            )
            print(f"running {pipeline_name} on file {input_file_path}")
            output_file_extension = pipeline['output_ext'] if 'output_ext' in pipeline.keys() else os.path.splitext(input_file_path)[1]
            output_file_path = os.path.join(
                output_dir,
                f"{os.path.splitext(input_file)[0]}{output_file_extension}"
            )
            if ("overwrite" not in pipeline.keys() or not pipeline["overwrite"]) and os.path.isfile(output_file_path):
                continue
            pipeline_impl.transform(
                config=pipeline,
                input_path=input_file_path,
                output_path=output_file_path
            )
        pipeline_impl.destroy(config=config)
        print(pipeline)


if __name__ == "__main__":
    import argparse
    import yaml
    config_path = os.getenv("local_config_path", "./configs/dev.yaml")
    parser = argparse.ArgumentParser(description=f'Run the pipelines from {config_path}.')\

    parser.add_argument('input_dir',  type=str)
    parser.add_argument('output_dir',  type=str)
    parser.add_argument('config_path',  type=str)
    args = parser.parse_args()
    print(f"loading config from {args.config_path}")
    assert os.path.isfile(args.config_path), f"No config found at {os.path.abspath(args.config_path)}"
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(f"Input dir", os.path.abspath(args.input_dir))
    config["pipelines"][0]["input_dir"]=args.input_dir
    print(f"Output dir", os.path.abspath(args.output_dir))
    config["pipelines"][-1]["output_dir"]=args.output_dir
    print(config)
    run_pipelines(config)
