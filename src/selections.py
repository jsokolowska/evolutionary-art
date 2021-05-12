class Selection:
    """
        Defines an interface for Selection classes
    """
    def select_one(self, population, fitness, maximize=False):
        pass


class BestFit(Selection):
    """
        Implements Best Fit selection strategy
    """

    def select_one(self, population, fitness, maximize=False):
        best = population[0]
        best_score = fitness(best)

        for member in population:
            score = fitness(member)
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best = member
                best_score = score

        return best
