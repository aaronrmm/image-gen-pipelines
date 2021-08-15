import json
import os
from typing import Dict

from art_pipelines.image_utils import load_image, save_image


def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i ** 2 for i in range(10000)])


class ImageGenerator:
    def __init__(self, loading_func: callable, generator_func: callable):
        self.generator_func = generator_func
        self.loaded_model = loading_func()

    def generate(self, output_dir, generation_config: Dict):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # save configs
        config_save_path = os.path.join(output_dir, "config.json")
        with open(config_save_path, "w") as save_config_file:
            json.dump(obj=generation_config, fp=save_config_file)

        # save init image
        if "init_image" in generation_config.keys():
            init_image = load_image(generation_config["init_image"])
            save_image(image=init_image, save_dir=output_dir, save_name="init_image")

        self.generator_func(**generation_config)
