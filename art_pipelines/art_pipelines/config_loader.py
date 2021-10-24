import json
import os
from typing import Dict

import yaml

config: Dict = None

def reload_config():
    try:
        global config
        with open(os.getenv("config_path","./configs/dev.yaml"), "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    except FileNotFoundError:
        print(f"Could not find a config yaml file at {os.path.abspath(os.getenv('config_path','./configs/dev.yaml'))}")


def print_config():
    print(yaml.dump(config, default_flow_style=False))


reload_config()
print_config()
