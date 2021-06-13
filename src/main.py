import numpy as np

import algorithm
import population
from fitness_functions import *
from aesthetic_functions import *
from tqdm import trange


import visualization


if __name__ == "__main__":

    width, height = 250, 250
    iterations = 150
    step = 30
    generation = 0
    show_images = False
    p = 10

    fitness1, index1 = random_aesthetic()
    fitness2, index2 = random_aesthetic()
    fitness3, index3 = random_aesthetic()
    print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))

    compound1 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0]])
    compound2 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0]])
    compound3 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0], random_aesthetic()[0]])
    weights = [1, 1, 1, 2]
    imitating_fitness = ImitationAesthetic("../target/starry_night_full.jpg", width=width, height=height, resize=True)
    fitness = ImageFitnessFunction(width, height, compound1, compound2, compound3, p, imitation=imitating_fitness, weights=weights)
    fitness_target = visualization.image_from_population(width, height, fitness.texture)
    fitness_target.save("../results/fitness_target.png")

    fitness_imitation = visualization.image_from_population(width, height, imitating_fitness.texture)
    fitness_imitation.save("../results/imitation.png")

    initial_population = population.generate_initial_population(width, height, [255, 255, 255])
    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.PSO(initial_population, 2, 2, 1, 0.4, 0.9, iterations, width, height)

    img = visualization.image_from_population(width, height, alg.best_location)
    img.save("../results/generation_{}.png".format(generation))

    if show_images:
        img.show("{} generation".format(generation))

    for generation in trange(step, iterations-1, step):
        alg.run(fitness, step)
        img = visualization.image_from_population(width, height, alg.best_location)
        img.save("../results/generation_{}.png".format(generation))

        if show_images:
            img.show("{} generation".format(generation))

    img = visualization.image_from_population(width, height, alg.best_location)
    img.save("../results/generation_{}.png".format(generation))
    if show_images:
        img.show("{} generation".format(generation))

    print("Finished after {} generations".format(generation))
