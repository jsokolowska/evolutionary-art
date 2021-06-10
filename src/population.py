import numpy as np
from typing import List


def generate_initial_population(width: int, height: int, color: List[int] = None):
    population = []

    for x in range(width):
        for y in range(height):
            if color is not None:
                population.append(list(color))
            else:
                population.append([0, 0, 0, 0])

    return np.array(population)
