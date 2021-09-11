import os
from PIL import Image

from art_pipelines.cv_types import FiletypeImage


def transform(config, input_path, output_path):
    to_type = FiletypeImage[config["to_type"]]
    print(to_type.name)
    original = Image.open(input_path)
    original.save(output_path, format=to_type.value["PIL"])
