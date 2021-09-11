import json
import os
from typing import Dict

import yaml

config: Dict = None


def reload_config():
    global config
    with open(os.getenv("config_path"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def print_config():
    print(yaml.dump(config, default_flow_style=False))


reload_config()
print_config()
