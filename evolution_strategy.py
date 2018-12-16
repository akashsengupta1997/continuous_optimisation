import numpy as np
import math
import time
from objective import schwefel_func


class EvolutionStrategy:
    def __init__(self, objective_func, num_control_vars, elitist=False):

        self.objective_func = objective_func
        self.num_control_vars = num_control_vars
        self.allowed_evals = 10000

        self.elitist = elitist

        self.num_parents = 2
        self.num_children = self.num_parents * 7

    def generate_intial_population(self):
        population = []
        for i in range(self.num_children):
            control_vars = 500*np.random.uniform(-1.0, 1.0, self.num_control_vars)
            covariance = np.random.randn(self.num_control_vars, self.num_control_vars)
            population.append([control_vars, covariance])

        return population

    def select_parents(self, children, previous_parents):
        if self.elitist:
            population = children + previous_parents
        else:
            population = children
        fvals = list(map(self.objective_func, [specimen[0] for specimen in population]))
        sorted_indices = np.argsort(fvals)
        parents = []
        for i in range(self.num_parents):
            parents.append(children[sorted_indices[i]])

        return parents

    def mutate_stratetgy_params(self, population):
        

    def mutate_control_variables(self, population):
        pass

    def recombine(self):
        pass



test = EvolutionStrategy(schwefel_func, 5)
pop = test.generate_intial_population()
print(test.select_parents(pop, None))