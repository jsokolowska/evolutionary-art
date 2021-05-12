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
