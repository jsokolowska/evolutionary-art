from PIL import Image
from numpy import asarray


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


class SimpleFitness(FitnessFunction):
    def __init__(self, color):
        self.color = color

    def __call__(self, color, x, y):
        r_diff = abs(color[0] - self.color[0])
        g_diff = abs(color[1] - self.color[1])
        b_diff = abs(color[2] - self.color[2])

        return r_diff + g_diff + b_diff


class AestheticFitness(FitnessFunction):
    def __init__(self, aesthetic, p, channel="c1"):
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
        self.data = asarray(artwork)
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
