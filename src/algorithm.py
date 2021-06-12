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


class PSO(Algorithm):
    """
        Implements modified Particle Swarm Optimization algorithm with adaptive inertia weight

        Instead of global neighbourhood Neumann topography is used
    """

    def __init__(self, population: List[List[int]], c1: float, c2: float, r, inertia_min: float, inertia_max: float, iterations: int, width: int, height: int):
        self.population = np.array(population)

        self.location = self.population
        self.best_location = copy.deepcopy(self.location)
        self.next_location = copy.deepcopy(self.location)
        self.velocity = np.array([(np.random.uniform(-255, 255), np.random.uniform(-255, 255), np.random.uniform(-255, 255)) for i in population])
        self.score = np.zeros(len(population))
        self.best_score = np.full(len(population), math.inf)

        self.c1 = c1
        self.c2 = c2
        self.range = r

        self.random_seed = np.random.uniform(0, 1)
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

        self.current_iteration = 0
        self.iterations = iterations

        self.width = width
        self.height = height

    def iteration(self, fitness):
        inertia_change = (self.current_iteration / self.iterations) * (self.inertia_max - self.inertia_min)
        inertia = self.inertia_max - inertia_change + self.random_seed

        self.score = fitness.evaluate_texture(self.location)

        for i, values in enumerate(zip(self.score, self.best_score, self.location, self.best_location), 0):
            score, best_score, location, best_location = values
            if score < best_score:
                self.best_score[i] = score
                self.best_location[i] = location

        random1 = self.c1 * np.random.uniform(0, 1, self.location.shape)
        random2 = self.c2 * np.random.uniform(0, 1, self.location.shape)

        best_self_velocity = random1 * (self.best_location - self.location)
        best_neighbour_velocity = random2 * (self.best_neighbours(self.range) - self.location)

        self.velocity = inertia * self.velocity + best_self_velocity + best_neighbour_velocity
        self.velocity = np.clip(self.velocity, -255, 255)

        self.next_location = self.location + self.velocity
        self.next_location = np.clip(self.next_location, 0, 255)

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

    def best_neighbour(self, x, y, r):
        best_index = -1
        best_score = math.inf

        # for i in range(-r, r):
        #     for j in range(-r, r):
        #         if i == 0 and j == 0:
        #             continue
        #
        #         if 0 <= x + i < self.width and 0 <= y + j < self.height:
        #             index = self.width * (y + j) + x + i
        #             if self.best_score[index] < best_score:
        #                 best_score = self.best_score[index]
        #                 best_index = index

        index = self.width * y + x - 1
        if 0 <= index < self.width * self.height and self.best_score[index] < best_score:
            best_score = self.best_score[index]
            best_index = index

        index = self.width * y + x + 1
        if 0 <= index < self.width * self.height and self.best_score[index] < best_score:
            best_score = self.best_score[index]
            best_index = index

        index = self.width * (y - 1) + x
        if 0 <= index < self.width * self.height and self.best_score[index] < best_score:
            best_score = self.best_score[index]
            best_index = index

        index = self.width * (y + 1) + x
        if 0 <= index < self.width * self.height and self.best_score[index] < best_score:
            best_index = index

        return self.best_location[best_index]

    def best_neighbours(self, r):
        x, y = 0, 0
        best_neighbours = copy.deepcopy(self.location)

        for i, location in enumerate(self.location, 0):
            best_neighbours[i] = self.best_neighbour(x, y, r)

            x += 1
            if x >= self.width:
                x = 0
                y += 1

        return best_neighbours
