import numpy as np

import algorithm
import population
import mutations
import selections
import fitness_functions
from aesthetic_functions import *
from tqdm import trange
from PIL import Image
from numpy import asarray

import visualization


class SimpleFitness(fitness_functions.FitnessFunction):
    def __init__(self, color):
        self.color = color

    def __call__(self, color, x, y):
        r_diff = abs(color[0] - self.color[0])
        g_diff = abs(color[1] - self.color[1])
        b_diff = abs(color[2] - self.color[2])

        return r_diff + g_diff + b_diff


class Temp(fitness_functions.FitnessFunction):
    def __call__(self, color, x, y):
        return 0


class AestheticFitness(fitness_functions.FitnessFunction):
    def __init__(self, aesthetic, p, channel="c1"):
        self.aesthetic = aesthetic
        self.p = p
        self.channel = channel

    def __call__(self, color, x, y):
        if self.channel == "c1":
            return abs(color[0] - self.aesthetic(x, y, self.p))
        elif self.channel == "c2":
            return abs(color[1] - self.aesthetic(x, y, self.p))
        elif self.channel == "c3":
            return abs(color[2] - self.aesthetic(x, y, self.p))
        else:
            raise RuntimeError("Invalid channel")


class ImitationAesthetic(fitness_functions.FitnessFunction):
    def __init__(self, artwork_path, channels=None):
        if channels is None:
            channels = ["c1", "c2", "c3"]
        artwork = Image.open(artwork_path)
        self.data = asarray(artwork)
        self.channels = channels

    def __call__(self, color, x, y):
        score = 0
        if "c1" in self.channels:
            score += abs(color[0] - self.data[y][x][0])
        if "c2" in self.channels:
            score += abs(color[1] - self.data[y][x][1])
        if "c3" in self.channels:
            score += abs(color[2] - self.data[y][x][2])
        return score


if __name__ == "__main__":

    width, height = 70, 70
    iterations = 150
    step = 10
    generation = 0
    show_images = False
    p = 10

    fitness1, index1 = random_aesthetic()
    fitness2, index2 = random_aesthetic()
    fitness3, index3 = random_aesthetic()
    print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))

    aesthetic_fitness_list = [AestheticFitness(fitness1, p, "c1"),
                              AestheticFitness(fitness2, p, "c2"),
                              AestheticFitness(fitness3, p, "c3"),
                              ImitationAesthetic("../target/starry_night.jpg")]

    weights = [1/6, 1/6, 1/6, 1]

    # fitness = SimpleFitness((255, 0, 255))
    fitness = fitness_functions.CompoundFitnessFunction(aesthetic_fitness_list, weights)
    # fitness = ImitationAesthetic("../target/drawisland.png")
    initial_population = population.generate_initial_population(width, height, [255, 255, 255, 0])
    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.PSO(initial_population, 2, 2, 0.4, 0.9, iterations, width, height)

    for i in trange(0, iterations, step):
        img = visualization.image_from_population(width, height, alg.population)
        img.save("../results/generation_{}.png".format(generation))

        print("Finished generation {}".format(generation))
        if show_images:
            img.show("{} generation".format(generation))

        alg.run(fitness, step)
        generation += step

        iterations -= step
        if iterations <= 0:
            break

    img = visualization.image_from_population(width, height, alg.population)
    img.save("../results/generation_{}.png".format(generation))
    if show_images:
        img.show("{} generation".format(generation))

    print("Finished after {} generations".format(generation))
