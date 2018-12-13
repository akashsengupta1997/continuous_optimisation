import numpy as np
import math
from objective import schwefel_func


class SimpleAnnealer:
    def __init__(self, objective_func, num_control_vars):
        self.objective_func = objective_func
        self.num_control_vars = num_control_vars

        # Max allowed change in each control variable for simple soln generator
        # TODO mess with sizes of diagonals
        self.C = np.identity(num_control_vars)

    def soln_generator(self, current_soln):
        """
        Generates new candidate solution from given current solution using Eqn 2 from simulated
        annealing handout: x_new = x_old + Cu.
        This is the simplest solution generation scheme, where C is just a constant diagonal
        matrix.
        New candidate solutions are forced to be between -500 and 500 inclusive.
        :param current_soln:
        :return:
        """
        u = np.random.uniform(-1.0, 1.0, (self.num_control_vars, 1))
        valid_solution = False
        while not valid_solution:
            new_soln = current_soln + np.dot(self.C, u)
            if np.all(new_soln <= 500) and np.all(new_soln >= -500):
                valid_solution = True
        return new_soln

    def acceptance_probability(self, current_f_val, new_fval, temperature):
        delta_f = new_fval - current_f_val

        if delta_f < 0:
            return 1
        else:
            acceptance_prob = math.exp(- delta_f / temperature)
            return acceptance_prob

    def simulated_annealing(self):

        current_soln = [0, 0]
        current_fval = self.objective_func(current_soln)
        best_soln = current_soln
        best_fval = current_fval
        temperature = 10

        for i in range(1000):
            new_soln = self.soln_generator(current_soln)
            new_fval = self.objective_func(new_soln)
            accept_prob = self.acceptance_probability(current_fval, new_fval, temperature)

            if accept_prob > np.random.rand():
                current_soln = new_soln
                current_fval = new_fval

                if current_soln < best_soln:
                    best_soln = current_soln.copy()
                    best_fval = current_fval

            temperature = 0.95*temperature








