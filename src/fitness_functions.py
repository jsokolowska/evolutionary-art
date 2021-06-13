import math

import numpy as np
from PIL import Image
from tqdm import trange
from scipy.ndimage.filters import gaussian_filter


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
        self.height = height
        self.imitation_texture = imitation.texture
        self.weights = weights if weights is not None else [1, 1, 1, 1.5]

    def evaluate_texture(self, texture):
        aesthetic_score = np.sum(self.weights[:3] * np.abs(texture - self.texture), 1)
        if self.imitation_texture is None:
            imitation_score = np.zeros(aesthetic_score.shape)
        else:
            imitation_score = np.sum(self.weights[-1] * np.abs(texture - self.imitation_texture), 1)
        return aesthetic_score + imitation_score

    def __call__(self, color, x, y):
        aesthetic_score = np.sum(self.weights * np.abs(self.texture[y * self.width + x] - color))
        imitation_score = self.weights[-1] * np.sum(
            np.abs(self.imitation_texture[y * self.width + x] - color)) if self.imitation_texture is not None else 0
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
    def __init__(self, artwork_path: str, width: int, height: int, resize=False):
        if width == 0 or height == 0:
            raise ValueError("Height and width must be provided")
        artwork = Image.open(artwork_path)
        if width > 0 and height > 0 and resize:
            artwork = artwork.resize((width, height))
        elif width > 0 and height > 0:
            artwork = artwork.crop(box=(0, 0, width, height))
        self.data = np.asarray(artwork)
        self.data = gaussian_filter(self.data, 1)
        self.texture = np.zeros((width * height, 3))

        print("Calculating imitation image")
        # radius = math.floor((width + height) / 2 * 0.01)
        for x in trange(width):
            for y in range(height):
                self.texture[y * width + x] = self.data[y][x]
        # radius = 1
        # for x in trange(width):
        #     for y in range(height):
        #         imitation_target = 0
        #         x_range = min(x + radius + 1, width) - max(x - radius, 0)
        #         y_range = min(y + radius + 1, width) - max(y - radius, 0)
        #         for j in range(max([x - radius, 0]), min([x + radius + 1, width])):
        #             for i in range(max([y - radius, 0]), min([y + radius + 1, width])):
        #                 imitation_factor = (x - j) ** 2 + (y - i) ** 2 + 1
        #                 imitation_target += self.data[y][x] / imitation_factor
        #         self.texture[y * width + x] = imitation_target / (x_range * y_range * 0.1)
        #         self.a = "a"

    def __call__(self, color, x, y):
        score = 0
        # for j in range(self.width):
        #     for k in range(self.height):
        #         impact = (x-j)**2 + (y-k)**2 + 1
        #         score += self.data[j][k] /impact
        # return score/self.size_factor
        return score

        # # self.texture = np.zeros((width * height, 3))
        # # self.color_factor = np.zeros((width * height, 3))
        # # imit_factor = 3 * width * height
        # # for x in range(width):
        # #     for y in range(height):
        # #         imitation_target = 0
        # #         c_factor = 0
        # #         for j in range(width):
        # #             for k in range(height):
        # #                 impact_j = (x - j) ** 2
        # #                 impact_k = (y - k) ** 2
        # #                 impact = impact_k + impact_j + 1
        # #                 imitation_target += self.data[j][k] / impact
        # #                 c_factor += 1 / impact
        # #         self.texture[y * width + x] = imitation_target / imit_factor
        # #         self.color_factor[y * width + x] = c_factor
        # # # self.texture /= 3 * width * height
        # # self.color_factor /= imit_factor
        #
        # return score
