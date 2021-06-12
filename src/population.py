import numpy as np
from typing import List


def generate_initial_population(width: int, height: int, color: List[int] = None):
    population = []

    for x in range(width):
        for y in range(height):
            if color is not None:
                population.append(list(color))
            else:
                c = np.random.randint(0, 256, size=4)
                c[3] = 0
                population.append(c)

    return np.array(population)
