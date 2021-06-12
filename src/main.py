import numpy as np

import algorithm
import population
from fitness_functions import *
from aesthetic_functions import *
from tqdm import trange


import visualization


if __name__ == "__main__":

    width, height = 400, 400
    iterations = 150
    step = 30
    generation = 0
    show_images = False
    p = 1

    fitness1, index1 = random_aesthetic()
    fitness2, index2 = random_aesthetic()
    fitness3, index3 = random_aesthetic()
    print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))

    aesthetic_fitness_list = [AestheticFitness(fitness1, width, height, p, "c1"),
                              AestheticFitness(fitness2, width, height, p, "c2"),
                              AestheticFitness(fitness3, width, height, p, "c3"),
                              ImitationAesthetic("../target/starry_night.jpg")]

    # weights = [1/6, 1/6, 1/6, 1]
    weights = [0.6, 0.6, 0.6, 1]
    # weights = [0, 0, 0, 1]
    imitation = np.asarray(Image.open("../target/scream.jpg").convert("RGB")).reshape((width * height, 3))

    # fitness = SimpleFitness((255, 0, 255))
    fitness = ImageFitnessFunction(width, height, fitness1, fitness2, fitness3, p, imitation.data, weights)
    fitness_target = visualization.image_from_population(width, height, fitness.texture)
    fitness_target.save("../results/fitness_target.png")

    # fitness = CompoundFitnessFunction(aesthetic_fitness_list, weights)
    # fitness = CompoundFitnessFunction([SimpleFitness((255, 0, 0)), SimpleFitness((0, 0, 255))])
    # fitness = ImitationAesthetic("../target/drawisland.png")
    initial_population = population.generate_initial_population(width, height, [255, 255, 255])
    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.PSO(initial_population, 2, 2, 0.9, 0.9, iterations, width, height)

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
