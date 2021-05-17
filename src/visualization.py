import numpy as np
from PIL import Image
from typing import List

from population import Individual


def population_to_texture(width: int, height: int, population: List[Individual]):
    texture = np.zeros((width, height, 3), np.int8)

    for member in population:
        x, y = member.x, member.y

        pixel = texture[y, x]
        pixel[0] = member.c1
        pixel[1] = member.c2
        pixel[2] = member.c3

    return texture


def image_from_texture(texture, color_mode="RGB"):
    img = Image.fromarray(texture, color_mode)

    return img


def image_from_population(width: int, height: int, population: List[Individual], color_mode="RGB"):
    texture = population_to_texture(width, height, population)
    img = Image.fromarray(texture, color_mode)

    return img
