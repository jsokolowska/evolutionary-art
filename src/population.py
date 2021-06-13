import numpy as np
from typing import List
from PIL import Image


def generate_initial_population(width: int, height: int, color: List[int] = None):
    population = []

    for x in range(width):
        for y in range(height):
            if color is not None:
                population.append(list(color))
            else:
                c = np.random.randint(0, 256, size=3)
                population.append(c)

    return np.array(population)


def from_image(path: str, width: int, height: int, clip=False):
    if width == 0 or height == 0:
        raise ValueError("Height and width must be provided")
    artwork = Image.open(path)
    artwork = artwork.convert("HSV")
    if width > 0 and height > 0 and not clip:
        artwork = artwork.resize((width, height))
    elif width > 0 and height > 0:
        artwork = artwork.crop(box=(0, 0, width, height))
    data = np.asarray(artwork)
    texture = np.zeros((width * height, 3))
    for x in range(width):
        for y in range(height):
            texture[y * width + x] = data[y][x]
    return texture
