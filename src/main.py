import algorithm
import population
import mutations
import selections
import fitness_functions

import visualization


class SimpleFitness(fitness_functions.FitnessFunction):
    def __init__(self, color):
        self.color = color

    def __call__(self, individual):
        r_diff = abs(individual.c1 - self.color[0])
        g_diff = abs(individual.c2 - self.color[1])
        b_diff = abs(individual.c3 - self.color[2])

        return r_diff + g_diff + b_diff


if __name__ == "__main__":

    width, height = 200, 200
    iterations = 200
    step = 10
    generation = 0
    show_images = False

    selection = selections.BestFit()
    mutation = mutations.GaussianMutation(0, 10)
    fitness = SimpleFitness((255, 0, 255))
    # fitness = fitness_functions.CompoundFitnessFunction([SimpleFitness((255, 0, 0)), SimpleFitness((0, 0, 255))])
    initial_population = population.generate_initial_population(width, height)
    print("Generated new population of {} individuals".format(width * height))

    alg = algorithm.BasicEvolution(initial_population, mutation, selection)

    while iterations > 0:
        img = visualization.image_from_population(width, height, alg.population)
        img.save("../results/generation_{}.png".format(generation))

        print("Finished generation {}".format(generation))
        if show_images:
            img.show("{} generation".format(generation))

        alg.run(fitness, step)
        generation += step

        iterations -= step

    img = visualization.image_from_population(width, height, alg.population)
    img.save("../results/generation_{}.png".format(generation))
    if show_images:
        img.show("{} generation".format(generation))

    print("Finished after {} generations".format(generation))
