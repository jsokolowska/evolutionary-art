import numpy as np
from PIL import Image
from typing import List

def population_to_texture(width: int, height: int, population):
    texture = np.zeros((width, height, 3), np.int8)

    x, y = 0, 0
    for member in population:
        pixel = texture[y, x]
        pixel[0] = member[0]
        pixel[1] = member[1]
        pixel[2] = member[2]

        x += 1
        if x >= width:
            x = 0
            y += 1

    return texture


def image_from_texture(texture, color_mode="RGB"):
    img = Image.fromarray(texture, color_mode)

    return img


def image_from_population(width: int, height: int, population, color_mode="RGB"):
    texture = population_to_texture(width, height, population)
    img = Image.fromarray(texture, color_mode)

    return img
