import numpy as np
from PIL import Image


class FitnessFunction:
    """
        Defines an interface for fitness functions
    """
    def __call__(self, color, x, y):
        pass


class CompoundFitnessFunction(FitnessFunction):
    """
        Evaluates multiple fitness functions according to specified weights
    """
    def __init__(self, functions, weights=None):
        self.functions = functions
        self.weights = weights if weights is not None else [1 for i in range(len(functions))]

    def __call__(self, color, x, y):
        score = 0
        for function, weight in zip(self.functions, self.weights):
            score += weight * function(color, x, y)

        return score


class ImageFitnessFunction(FitnessFunction):
    def __init__(self, width, height, c1, c2, c3, p, imitation=None, weights=None):
        self.texture = np.zeros((width * height, 3))
        for x in range(width):
            for y in range(height):
                self.texture[y * width + x] = [c1(x, y, p), c2(x, y, p), c3(x, y, p)]

        self.width = width
        self.imitation = imitation
        self.weights = weights if weights is not None else [1, 1, 1, 1]

    def evaluate_texture(self, texture):
        aesthetic_score = np.sum(self.weights[:3] * np.abs(texture - self.texture), 1)
        imitation_score = self.weights[-1] * np.sum(np.abs(texture - self.imitation), 1) if self.imitation is not None else np.zeros(aesthetic_score.shape)
        return aesthetic_score + imitation_score

    def __call__(self, color, x, y):
        aesthetic_score = np.sum(self.weights * np.abs(self.texture[y * self.width + x] - color))
        imitation_score = self.weights[-1] * np.sum(np.abs(self.imitation[y * self.width + x] - color)) if self.imitation is not None else 0
        return aesthetic_score + imitation_score


class SimpleFitness(FitnessFunction):
    def __init__(self, color):
        self.color = color

    def __call__(self, color, x, y):
        r_diff = abs(color[0] - self.color[0])
        g_diff = abs(color[1] - self.color[1])
        b_diff = abs(color[2] - self.color[2])

        return r_diff + g_diff + b_diff


class AestheticFitness(FitnessFunction):
    def __init__(self, aesthetic, width, height, p, channel="c1"):
        self.aesthetic = aesthetic
        self.p = p
        self.channel = channel

    def __call__(self, color, x, y):
        if self.channel == "c1":
            return abs(color[0] - self.aesthetic(x, y, self.p))
        elif self.channel == "c2":
            return abs(color[1] - self.aesthetic(x, y, self.p))
        elif self.channel == "c3":
            return abs(color[2] - self.aesthetic(x, y, self.p))
        else:
            raise RuntimeError("Invalid channel")


class ImitationAesthetic(FitnessFunction):
    def __init__(self, artwork_path, channels=None):
        if channels is None:
            channels = ["c1", "c2", "c3"]
        artwork = Image.open(artwork_path)
        self.data = np.asarray(artwork)
        self.channels = channels

    def __call__(self, color, x, y):
        score = 0
        if "c1" in self.channels:
            score += abs(color[0] - self.data[y][x][0])
        if "c2" in self.channels:
            score += abs(color[1] - self.data[y][x][1])
        if "c3" in self.channels:
            score += abs(color[2] - self.data[y][x][2])
        return score
