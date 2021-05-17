import copy
import math

import numpy as np
from typing import List

from population import Individual
from mutations import Mutation
from selections import Selection
from fitness_functions import FitnessFunction


class Algorithm:
    """
        Defines an interface for evolutionary algorithm
    """

    def iteration(self, fitness):
        pass

    def run(self, fitness, iterations):
        pass


class BasicEvolution(Algorithm):
    """
        Implements 1+1 evolution strategy for each member of population

        Each population member generates exactly one child that undergoes a mutation process
        Then, for each individual selection is performed between parent and his child
    """

    def __init__(self, population: List[Individual], mutation: Mutation, selection: Selection):
        self.population = np.array(population)
        self.next_generation = np.empty(len(self.population), Individual)

        self.mutation = mutation
        self.selection = selection

    def iteration(self, fitness: FitnessFunction):
        for i, member in enumerate(self.population, 0):
            offspring = self.mutation.mutate(member)
            best = self.selection.select_one([member, offspring], fitness)

            self.next_generation[i] = best

        self.population = self.next_generation

    def run(self, fitness: FitnessFunction, iterations=100):
        while iterations > 0:
            self.iteration(fitness)
            iterations -= 1


class PSO(Algorithm):
    """
        Implements modified Particle Swarm Optimization algorithm with adaptive inertia weight

        Instead of global neighbourhood Neumann topography is used
    """

    def __init__(self, population: List[Individual], c1: float, c2: float, inertia_min: float, inertia_max: float, iterations: int, width: int, height: int):
        self.population = np.array(population)

        self.location = self.population
        self.best_location = copy.deepcopy(self.population)
        self.velocity = np.array([np.array((np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1))) for member in self.population])

        self.c1 = c1
        self.c2 = c2

        self.r = np.random.uniform(0, 1)
        self.inertia = inertia_max
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

        self.k = 0
        self.iterations = iterations

        self.width = width
        self.height = height

    def iteration(self, fitness):
        for i, location in enumerate(self.location):
            best_self_velocity = self.c1 * np.random.uniform(0, 1) * np.array(self.best_location[i] - location)
            best_neighbour_velocity = self.c2 * np.random.uniform(0, 1) * np.array(self.best_neighbour(fitness, location) - location)

            self.velocity[i] += self.inertia * self.velocity[i] + best_self_velocity + best_neighbour_velocity
            self.velocity[i] = np.clip(self.velocity[i], -255, 255)

            self.location[i] += tuple(self.velocity[i])
            self.location[i].c1 = np.clip(self.location[i].c1, 0, 255)
            self.location[i].c2 = np.clip(self.location[i].c2, 0, 255)
            self.location[i].c3 = np.clip(self.location[i].c3, 0, 255)

            if fitness(self.location[i]) < fitness(self.best_location[i]):
                self.best_location[i] = self.location[i]

        self.k += 1
        self.inertia = self.inertia_max - (self.k / self.iterations) * (self.inertia_max - self.inertia_min) + self.r

    def run(self, fitness: FitnessFunction, iterations=100):
        while iterations > 0:
            self.iteration(fitness)
            iterations -= 1

        self.population = self.location

    def best_neighbour(self, fitness, location):
        best = None
        best_score = math.inf

        for i in range(-1, 2):
            x = location.x + i
            if 0 <= x < self.width:
                for j in range(-1, 2):
                    y = location.y + j
                    if (x != 0 and y != 0) and 0 <= y < self.height:
                        score = fitness(self.location[x + self.width * y])
                        if score < best_score:
                            best = self.location[x + self.width * y]
                            best_score = score

        return best
