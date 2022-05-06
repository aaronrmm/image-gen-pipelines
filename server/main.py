import importlib
import os
import sys

import numpy as np

lib_path = "./art_pipelines/art_pipelines/pipelines"
assert os.path.isdir(lib_path)
sys.path.append(lib_path)


import typing

import PIL.Image
from fastapi import FastAPI
from tqdm.auto import tqdm
import base64

app = FastAPI()
from PIL import Image
import io

from pydantic import BaseModel

class Input(BaseModel):
    image_data: str
    config: typing.Any

@app.post("/run")
async def root(data:Input):
    image_data = data.image_data
    if True:
        f = io.BytesIO(base64.b64decode(image_data))
        image = Image.open(f)
        results: typing.List[Image] = run_pipelines_in_memory(images=[image], config=data.config)
        result_b64strs: typing.List[str] = [
            image_to_b64str(np2image(result))
            for result in results
        ]
        return {"results": result_b64strs, "config":data.config}
    #except:
    #    return {"message": "failure", "config":data.config}

def np2image(array: np.ndarray):
    return PIL.Image.fromarray(array)

def image_to_b64str(input_image: Image)->str:
    buf = io.BytesIO()
    input_image.save(buf, format='BMP')
    input_bytes = buf.getvalue()
    input_encoded = base64.b64encode(input_bytes)
    input_ascii = input_encoded.decode('ascii')
    return input_ascii

def run_pipelines_in_memory(images: typing.List[PIL.Image.Image], config: typing.Dict)-> typing.List[PIL.Image.Image]:
    for pipeline in config["pipelines"]:
        if "skip" in pipeline and pipeline["skip"]:
            continue
        pipeline_name = pipeline["name"]
        print("Loading pipeline impl ", pipeline_name)
        pipeline_impl = importlib.import_module(
            f"{pipeline_name}.inmem_implementation", package=".."
        )
        print("Loaded pipeline impl ", pipeline_name)
        pipeline_impl.load(config=pipeline)
        images = pipeline_impl.transform(
            config=pipeline,
            images=images,
        )
        pipeline_impl.destroy(config=config)
    return images