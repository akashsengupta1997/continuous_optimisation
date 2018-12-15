import numpy as np
import math


class SimpleAnnealer:
    def __init__(self, objective_func, num_control_vars):
        self.objective_func = objective_func
        self.num_control_vars = num_control_vars
        self.allowed_evals = 10000
        self.initial_temperature_search_evals = 700

        # Max allowed change in each control variable for simple soln generator
        self.C = 200*np.identity(num_control_vars)
        self.alpha = 0.95  # Temperature decrement multiplier
        self.chain_length = 10  # Number of iterations at each temperature

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
        valid_solution = False
        while not valid_solution:
            u = np.random.uniform(-1.0, 1.0, self.num_control_vars)
            new_soln = current_soln + np.dot(self.C, u)
            if np.all(new_soln <= 500) and np.all(new_soln >= -500):
                valid_solution = True
        return new_soln

    def random_soln_generator(self):
        return 500*np.random.uniform(-1.0, 1.0, self.num_control_vars)

    def acceptance_probability(self, current_f_val, new_fval, temperature):
        delta_f = new_fval - current_f_val

        if delta_f < 0:
            return 1
        else:
            acceptance_prob = math.exp(- delta_f / temperature)
            return acceptance_prob

    def initial_temperature_search(self, initial_acceptance_prob):
        current_soln = self.random_soln_generator()
        print('init search init soln', current_soln)
        current_fval = self.objective_func(current_soln)
        fval_increases = []

        for i in range(self.initial_temperature_search_evals - 1):
            # need -1 here because doing 1 function evaluation outside the loop
            new_soln = self.soln_generator(current_soln)
            new_fval = self.objective_func(current_soln)
            delta_f = new_fval - current_fval
            if delta_f > 0:
                fval_increases.append(delta_f)
            current_soln = new_soln.copy()
            current_fval = new_fval

        initial_temperature = - np.mean(fval_increases) / math.log(initial_acceptance_prob)
        # print('mean fval increase', np.mean(fval_increases))
        # print('T0', initial_temperature)
        return initial_temperature

    def simulated_annealing(self):

        current_soln = self.random_soln_generator()
        print('init soln', current_soln)
        current_fval = self.objective_func(current_soln)
        best_soln = current_soln
        best_fval = current_fval
        temperature = self.initial_temperature_search(0.8)
        solns = []
        fvals = []
        num_trials_current_temperature = 0
        num_acceptances_current_temperature = 0

        for i in range(self.allowed_evals - self.initial_temperature_search_evals - 1):
            # need -1 here because doing 1 function evaluation outside the loop
            new_soln = self.soln_generator(current_soln)
            new_fval = self.objective_func(new_soln)
            accept_prob = self.acceptance_probability(current_fval, new_fval, temperature)
            num_trials_current_temperature += 1

            if accept_prob > np.random.rand():
                current_soln = new_soln.copy()
                current_fval = new_fval
                solns.append(current_soln)
                fvals.append(current_fval)
                num_acceptances_current_temperature += 1

                if current_fval < best_fval:
                    best_soln = current_soln.copy()
                    best_fval = current_fval

            # Adjust temperature if passed self.chain_length trials at current temperature or
            # if self.chain_length*0.6 acceptances have occurred at current temperature
            if num_trials_current_temperature > self.chain_length or \
                    num_acceptances_current_temperature > self.chain_length * 0.6:
                temperature = self.alpha * temperature
                num_trials_current_temperature = 0
                num_acceptances_current_temperature = 0

        return best_soln, best_fval, solns, fvals








