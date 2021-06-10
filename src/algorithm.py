import copy
import math

import numpy as np
from typing import List

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


# class BasicEvolution(Algorithm):
#     """
#         Implements 1+1 evolution strategy for each member of population
#
#         Each population member generates exactly one child that undergoes a mutation process
#         Then, for each individual selection is performed between parent and his child
#     """
#
#     def __init__(self, population: List[Individual], mutation: Mutation, selection: Selection):
#         self.population = np.array(population)
#         self.next_generation = np.empty(len(self.population), Individual)
#
#         self.mutation = mutation
#         self.selection = selection
#
#     def iteration(self, fitness: FitnessFunction):
#         for i, member in enumerate(self.population, 0):
#             offspring = self.mutation.mutate(member)
#             best = self.selection.select_one([member, offspring], fitness)
#
#             self.next_generation[i] = best
#
#         self.population = self.next_generation
#
#     def run(self, fitness: FitnessFunction, iterations=100):
#         while iterations > 0:
#             self.iteration(fitness)
#             iterations -= 1


class PSO(Algorithm):
    """
        Implements modified Particle Swarm Optimization algorithm with adaptive inertia weight

        Instead of global neighbourhood Neumann topography is used
    """

    def __init__(self, population: List[List[int]], c1: float, c2: float, inertia_min: float, inertia_max: float, iterations: int, width: int, height: int):
        self.population = np.array(population)

        self.location = self.population
        self.best_location = copy.deepcopy(self.location)
        self.next_location = copy.deepcopy(self.location)
        self.velocity = np.array([(np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0) for i in population])
        self.score = np.zeros(len(population))

        self.c1 = c1
        self.c2 = c2

        self.random_seed = np.random.uniform(0, 1)
        self.inertia = inertia_max
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

        self.current_iteration = 0
        self.iterations = iterations

        self.width = width
        self.height = height

    def iteration(self, fitness):
        inertia_change = (self.current_iteration / self.iterations) * (self.inertia_max - self.inertia_min)
        self.inertia = self.inertia_max - inertia_change + self.random_seed

        self.evaluate_particles(fitness)

        x, y = 0, 0

        for i, location in enumerate(self.location, 0):
            best_self_velocity = self.c1 * np.random.uniform(0, 1) * np.array(self.best_location[i] - location)
            best_neighbour_velocity = self.c2 * np.random.uniform(0, 1) * np.array(self.best_neighbour(x, y) - location)

            self.velocity[i] += self.inertia * self.velocity[i] + best_self_velocity + best_neighbour_velocity
            self.velocity[i] = np.clip(self.velocity[i], -255, 255)

            self.next_location[i] = self.location[i] + tuple(self.velocity[i])

            self.next_location[i] = np.clip(self.next_location[i], 0, 255)

            if fitness(self.next_location[i], x, y) < fitness(self.best_location[i], x, y):
                self.best_location[i] = self.next_location[i]

            x += 1
            if x >= self.width:
                x = 0
                y += 1

        temp = self.location
        self.location = self.next_location
        self.next_location = temp

        self.current_iteration += 1

    def run(self, fitness: FitnessFunction, iterations=100):
        while iterations > 0:
            self.iteration(fitness)
            iterations -= 1

        self.population = self.location

    def evaluate_particles(self, fitness):
        x, y = 0, 0
        for i, location in enumerate(self.location, 0):
            self.score[i] = fitness(location, x, y)

            x += 1
            if x >= self.width:
                x = 0
                y += 1

    def best_neighbour(self, x, y):
        best_index = -1
        best_score = math.inf

        index = self.width * y + x - 1
        if 0 <= index < self.width * self.height and self.score[index] < best_score:
            best_score = self.score[index]
            best_index = index

        index = self.width * y + x + 1
        if 0 <= index < self.width * self.height and self.score[index] < best_score:
            best_score = self.score[index]
            best_index = index

        index = self.width * (y - 1) + x
        if 0 <= index < self.width * self.height and self.score[index] < best_score:
            best_score = self.score[index]
            best_index = index

        index = self.width * (y + 1) + x
        if 0 <= index < self.width * self.height and self.score[index] < best_score:
            best_index = index

        return self.location[best_index]
