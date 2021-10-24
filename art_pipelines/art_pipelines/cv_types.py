from enum import Enum


class FiletypeImage(Enum):
    png = {"name": "png", "PIL": "png"}
    jpg = {"name": "jpg", "PIL": "jpeg"}
