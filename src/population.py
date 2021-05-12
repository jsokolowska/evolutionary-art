import numpy as np


class Individual:
    def __init__(self, x, y, c1=None, c2=None, c3=None):
        self.x = x
        self.y = y

        self.c1 = c1 if c1 is not None else np.random.randint(0, 256)
        self.c2 = c2 if c2 is not None else np.random.randint(0, 256)
        self.c3 = c3 if c3 is not None else np.random.randint(0, 256)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "x: {}, y: {}, c1: {}, c2: {}, c3: {}".format(self.x, self.y, self.c1, self.c2, self.c3)


def generate_initial_population(width: int, height: int):
    population = []

    for x in range(width):
        for y in range(height):
            population.append(Individual(x, y))

    return np.array(population)
