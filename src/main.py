import numpy as np

import algorithm
import population
from fitness_functions import *
from aesthetic_functions import *
from tqdm import trange
from argparse import Namespace

import visualization

#
#     width, height = 400, 400
#     iterations = 30
#     step = 5
#     generation = 0
#     show_images = False
#     p = 10
#
#     fitness1, index1 = random_aesthetic()
#     fitness2, index2 = random_aesthetic()
#     fitness3, index3 = random_aesthetic()
#     print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))
#
#     compound1 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0]])
#     compound2 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0]])
#     compound3 = CompoundFitnessFunction([random_aesthetic()[0], random_aesthetic()[0]])
#     weights = [1, 1, 1, 0]
#     imitating_fitness = ImitationAesthetic("../target/starry_night_full.jpg", width=width, height=height, resize=True)
#     fitness = ImageFitnessFunction(width, height, compound1, compound2, compound3, p, imitation=imitating_fitness,
#                                    weights=weights)
#     fitness_target = visualization.image_from_population(width, height, fitness.texture)
#     fitness_target.save("../results/fitness_target.png")
#
#     fitness_imitation = visualization.image_from_population(width, height, imitating_fitness.texture)
#     fitness_imitation.save("../results/imitation.png")
#
#     # initial_population = population.generate_initial_population(width, height, [0, 0, 255])
#     initial_population = population.from_image("../target/colors.png", width=width, height=height, resize=True)
#     print("Generated new population of {} individuals".format(width * height))
#
#     alg = algorithm.PSO(initial_population, 2, 2, 10, 0.4, 0.9, iterations, width, height)
#
#     img = visualization.image_from_population(width, height, alg.location)
#     img.save("../results/generation_{}.png".format(generation))
#
#     if show_images:
#         img.show("{} generation".format(generation))
#
#     for generation in trange(step, iterations + 1, step):
#         alg.run(fitness, step)
#         img = visualization.image_from_population(width, height, alg.location)
#         img.save("../results/generation_{}.png".format(generation))
#
#         if show_images:
#             img.show("{} generation".format(generation))
#
#     img = visualization.image_from_population(width, height, alg.best_location)
#     img.save("../results/generation_{}.png".format(generation))
#     if show_images:
#         img.show("{} generation".format(generation))
#
#     print("Finished after {} generations".format(generation))


def init_and_run(args):
    # initial population
    width, height = args.width, args.height
    if args.input is not None:
        initial_population = population.from_image(args.input, width, height)
    else:
        initial_population = population.generate_initial_population(width, height, [0, 0, 255])

    iterations = args.iterations
    step = args.step
    show_images = args.show
    p = args.p

    fitness1, index1 = random_aesthetic()
    fitness2, index2 = random_aesthetic()
    fitness3, index3 = random_aesthetic()
    print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))

    weights = args.weights
    imitating_fitness = None
    if args.target is not None:
        imitating_fitness = ImitationAesthetic(args.target, width=width, height=height, resize=True)
    fitness = ImageFitnessFunction(width, height, fitness1, fitness2, fitness3, p, imitation=imitating_fitness,
                                   weights=weights)
    fitness_target = visualization.image_from_population(width, height, fitness.texture)
    fitness_target.save("../results/fitness_target.png")

    if imitating_fitness is not None :
        fitness_imitation = visualization.image_from_population(width, height, imitating_fitness.texture)
        fitness_imitation.save("../results/imitation.png")

    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.PSO(initial_population, args.c1, args.c2, args.max_velocity, args.inertia_min,
                        args.inertia_max, iterations, width, height)

    img = visualization.image_from_population(width, height, alg.location)
    generation = 0
    img.save("../results/generation_{}.png".format(generation))

    if show_images:
        img.show("{} generation".format(generation))

    for generation in trange(step, iterations + 1, step):
        alg.run(fitness, step)
        img = visualization.image_from_population(width, height, alg.location)
        img.save("../results/generation_{}.png".format(generation))

        if show_images:
            img.show("{} generation".format(generation))


if __name__ == "__main__":
    args = Namespace()
    args.width = 400
    args.height = 400
    args.iterations = 150
    args.step = 30
    args.show = False
    args.input = None
    args.functions = 1
    args.weights = [1, 1, 1, 1.3]
    args.target = None
    args.imitation_mode = 'resize'
    args.p = 10
    args.c1 = 2
    args.c2 = 2
    args.max_velocity = 10
    args.inertia_min = 0.4
    args.inertia_max = 0.9
    args.input = '../target/colors.png'

    init_and_run(args)
