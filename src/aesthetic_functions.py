import math
import numpy as np


def aesthetic1(x: int, y: int, p: int):
    return (x | y) % 255


def aesthetic2(x: int, y: int, p: int):
    return (p & y) % 255


def aesthetic3(x: int, y: int, p: int):
    return (x / (1 + y + p)) % 255


def aesthetic4(x: int, y: int, p: int):
    return (x * y) % 255


def aesthetic5(x: int, y: int, p: int):
    return (x + y) % 255


def aesthetic6(x: int, y: int, p: int):
    return abs(x - y) % 255


def aesthetic7(x: int, y: int, p: int):
    return 255 - (x % 255)


def aesthetic8(x: int, y: int, p: int):
    return abs(255 * math.cos(x))


def aesthetic9(x: int, y: int, p: int):
    return abs(255 * math.tan((x % 45) * math.pi / 180))


def aesthetic10(x: int, y: int, p: int):
    return abs((255 * math.tan(x)) % 255)


def random_aesthetic():
    aesthetic_functions = [aesthetic1, aesthetic2, aesthetic3, aesthetic4, aesthetic5, aesthetic6, aesthetic7,
                           aesthetic8, aesthetic9, aesthetic10]

    index = np.random.randint(0, len(aesthetic_functions))

    return aesthetic_functions[index], index + 1
