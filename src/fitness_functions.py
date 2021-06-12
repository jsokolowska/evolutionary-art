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