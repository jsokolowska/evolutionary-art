import math
import numpy as np

def discrete_a1(x, y, p):
    return ((x | y) / p) % 255


def aesthetic1(x: int, y: int, p: int):
    return (x | y) % 255


def aesthetic2(x: int, y: int, p: int):
    return (p & y) % 255


def aesthetic3(x: float, y: float, p: float):
    return (x / (1 + y + p)) % 255


def aesthetic4(x: float, y: float, p: float):
    return (x * y) % 255


def aesthetic5(x: float, y: float, p: float):
    return (x + y) % 255


def aesthetic6(x: float, y: float, p: float):
    return abs(x - y) % 255


def aesthetic7(x: float, y: float, p: float):
    return 255 - (x % 255)


def aesthetic8(x: float, y: float, p: float):
    return abs(255 * math.cos(x))


def aesthetic9(x: float, y: float, p: float):
    return abs(255 * math.tan((x % 45) * math.pi / 180))


def aesthetic10(x: float, y: float, p: float):
    return abs((255 * math.tan(x)) % 255)


def aesthetic11(x: float, y: float, p: float):
    return math.sqrt(x*x + y*y) % 255


def aesthetic12(x: float, y: float, p: float):
    return x % (p + 1) + 255 - p


def aesthetic13(x: float, y: float, p: float):
    return ((x + y) / 2) % 255


def aesthetic14(x: float, y: float, p: float):
    return 255 * ((x + 1) / (y + 1)) if x < y else 255 * ((y + 1) / (x + 1))


def aesthetic15(x: float, y: float, p: float):
    return math.sqrt(abs(x*x + y*y - 2*p*p)) % 255


def random_aesthetic():
    aesthetic_functions = [aesthetic1, aesthetic2, aesthetic3, aesthetic4, aesthetic5, aesthetic6, aesthetic7,
                           aesthetic8, aesthetic9, aesthetic10, aesthetic11, aesthetic12, aesthetic13, aesthetic14,
                           aesthetic15]

    index = np.random.randint(0, len(aesthetic_functions))

    return aesthetic_functions[index], index + 1


