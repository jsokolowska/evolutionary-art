import numpy as np

import algorithm
import population
from fitness_functions import *
from aesthetic_functions import *
from tqdm import trange
from argparse import Namespace

import visualization


def init_and_run(arguments):
    # initial population
    width, height = arguments.width, arguments.height
    if arguments.input is not None:
        initial_population = population.from_image(arguments.input, width, height, arguments.clip_input)
    else:
        initial_population = population.generate_initial_population(width, height, [0, 0, 255])

    iterations = arguments.iterations
    step = arguments.step
    show_images = arguments.show
    p = arguments.p

    fitness1, index1 = random_aesthetic()
    fitness2, index2 = random_aesthetic()
    fitness3, index3 = random_aesthetic()
    print("Aesthetic functions chosen: c1-{}, c2-{}, c3-{}".format(index1, index2, index3))

    weights = arguments.weights
    imitating_fitness = None
    if arguments.target is not None:
        imitating_fitness = ImitationAesthetic(arguments.target, width=width, height=height,
                                               clip=args.clip_target)
    fitness = ImageFitnessFunction(width, height, aesthetic12, aesthetic9, aesthetic10, p, imitation=imitating_fitness,
                                   weights=weights)
    fitness_target = visualization.image_from_population(width, height, fitness.texture)
    fitness_target.save("../results/fitness_target.png")

    if imitating_fitness is not None:
        fitness_imitation = visualization.image_from_population(width, height, imitating_fitness.texture)
        fitness_imitation.save("../results/imitation.png")

    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.PSO(initial_population, arguments.c1, arguments.c2, arguments.max_velocity, arguments.inertia_min,
                        arguments.inertia_max, iterations, width, height, arguments.radius)

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
    args.radius = 4
    args.clip_target = False
    args.clip_input = False

    init_and_run(args)
