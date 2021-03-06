# https://github.com/ChristophReich1996/Swin-Transformer-V2

from pipelines.super_resolution.magnifier import Magnifier

singleton: Magnifier


def load(config):
    global singleton
    print("loading...")
    print(config.keys())
    if "model_name" in config and config["model_name"] not in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:
        if config["model_name"].lower() in ["bsrgan"]:
            from .bsrgan import BsrganMagnifier
            singleton = BsrganMagnifier()
        elif config["model_name"].lower() in ["swinir"]:
            from .swinIR import SwinIrMagnifier
            singleton = SwinIrMagnifier()
        else:
            raise NotImplementedError(f"unknown model name{config['model_name']}")
    elif config["model_name"]:  # ['RealESRGAN_x4plus', 'RealESRNet_x4plus']
        from .bsrgan import BsrganMagnifier
        singleton = BsrganMagnifier()
    singleton.load(config)
    print("Finished loading!")


def transform(config, input_path, output_path):
    global singleton
    print("transforming...")
    singleton.transform(config, input_path, output_path)


def destroy(config):
    global singleton
    singleton = None
