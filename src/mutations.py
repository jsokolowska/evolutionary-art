import copy
import numpy as np


class Mutation:
    """
        Defines interface for mutation classes
    """

    def mutate(self, individual):
        pass


class GaussianMutation(Mutation):
    def __init__(self, mean, deviation):
        self.mean = mean
        self.deviation = deviation

    def mutate(self, individual):
        new_object = copy.deepcopy(individual)

        new_object.c1 = np.clip(int(individual.c1 + np.random.normal(self.mean, self.deviation)), 0, 255)
        new_object.c2 = np.clip(int(individual.c2 + np.random.normal(self.mean, self.deviation)), 0, 255)
        new_object.c3 = np.clip(int(individual.c3 + np.random.normal(self.mean, self.deviation)), 0, 255)

        return new_object
