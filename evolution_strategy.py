import numpy as np
import math
import time
from objective import schwefel_func

# TODO implement global recombination


class EvolutionStrategy:
    def __init__(self, objective_func, num_control_vars, elitist=False,
                 global_recombination=False):

        self.objective_func = objective_func
        self.num_control_vars = num_control_vars
        self.allowed_evals = 5000

        self.elitist = elitist
        self.global_recombination = global_recombination

        self.num_parents = 10
        self.num_children = self.num_parents * 7
        self.recombination_weight = 0.5

    def generate_intial_population(self):
        population = []
        for i in range(self.num_children):
            control_vars = 500*np.random.uniform(-1.0, 1.0, self.num_control_vars)
            stds = np.sqrt(np.random.randn(self.num_control_vars)**2)
            temp = np.random.randn(self.num_control_vars, self.num_control_vars)
            rot_angles = (temp - temp.T)/2  # skew-symmetric with diagonals = 0
            population.append([control_vars, stds, rot_angles])

        return population

    def select_parents(self, children, previous_parents):
        """

        :param children:
        :param previous_parents:
        :return:
        """
        if self.elitist and previous_parents is not None:
            population = children + previous_parents
        else:
            population = children

        # Check that all control variables within bounds - remove solutions with invalid
        # control variables from population
        invalid_indices = []
        for i in range(len(population)):
            control_vars = population[i][0]
            if np.any(control_vars > 500):
                invalid_indices.append(i)

        # Assess population
        # Using map is a little bit faster than appending in for loop
        fvals = list(map(self.objective_func, [solution[0] for solution in population]))
        sorted_indices = np.argsort(fvals)
        # Remove invalid indices from sorted_indices list
        for index in invalid_indices:
            sorted_indices = list(sorted_indices)
            sorted_indices.remove(index)

        # Select top num_parents solutions as new parents
        parents = []
        for i in range(self.num_parents):
            parents.append(population[sorted_indices[i]])

        num_func_evals = len(fvals)
        children_fvals = fvals[:self.num_children]

        return parents, num_func_evals, children_fvals

    def construct_cov_matrix(self, rotation_angles, stds):
        """
        Diagonals are incorrect - actual diagonals are stds.
        :param rotation_angles:
        :return:
        """
        cov_matrix = np.zeros((self.num_control_vars, self.num_control_vars))
        for i in range(self.num_control_vars):
            for j in range(self.num_control_vars):
                if i == j:
                    cov_matrix[i, j] = stds[i] ** 2
                else:
                    cov_matrix[i, j] = 0.5 * (stds[i] ** 2 - stds[j] ** 2) * np.tan(
                        2 * rotation_angles[i, j])

        return cov_matrix

    def mutate_stratetgy_params(self, solution):
        tau = 1/math.sqrt(2*math.sqrt(self.num_control_vars))
        tau_prime = 1/math.sqrt(2*self.num_control_vars)
        beta = 0.0873

        chi_0 = np.random.randn()
        chi_i = np.random.randn(self.num_control_vars)
        temp = np.random.randn(self.num_control_vars, self.num_control_vars)
        chi_ij = (temp - temp.T)/2  # skew-symmetric

        stds = solution[1]
        new_stds = np.multiply(stds, np.exp(tau_prime * chi_0 + tau * chi_i))

        # For rotation angle matrices, only off-diagonal terms are relevant
        rot_angles = solution[2]
        new_rot_angles = rot_angles + beta * chi_ij
        # print(rot_angles)
        # print(new_rot_angles)
        # print(np.absolute(rot_angles)>np.pi)
        # print(np.absolute(new_rot_angles)>np.pi)

        return new_stds, new_rot_angles

    def mutate_solutions(self, parents):
        """
        Mutate parents before recombination.
        :param parents:
        :return:
        """
        mutated_population = []
        for solution in parents:
            new_stds, new_rot_angles = self.mutate_stratetgy_params(solution)

            # Cov_matrix should be symmetric (and rot_angles matrix is skew-symmetric)
            cov_matrix = self.construct_cov_matrix(new_rot_angles, new_stds)
            n = np.random.multivariate_normal(np.zeros(self.num_control_vars), cov_matrix,
                                              check_valid='warn')
            new_control_vars = solution[0] + n
            mutated_population.append([new_control_vars, new_stds, new_rot_angles])
        return mutated_population

    def control_var_recombination(self, parent_control_vars1, parent_control_vars2):
        """

        :param parent_control_vars: control varaibles of 2 randomly sampled parents
        :return:
        """
        # Discrete recombination
        cross_points = np.random.rand(self.num_control_vars) < 0.5  # p(cross) = 0.5 (fair coin toss)
        child_control_vars = np.where(cross_points, parent_control_vars1, parent_control_vars2)

        return child_control_vars

    def global_recombinator(self, parents):
        child_control_vars = []
        child_stds = []
        child_rot_angle = np.zeros((self.num_control_vars, self.num_control_vars))
        fixed = np.zeros((self.num_control_vars, self.num_control_vars), dtype=bool)

        for i in range(self.num_control_vars):
            parent_choice_cv = np.random.randint(0, self.num_parents)
            child_control_vars.append(parents[parent_choice_cv][0][i])
            parent_choice_std = np.random.randint(0, self.num_parents)
            child_stds.append(parents[parent_choice_std][1][i])

        for i in range(self.num_control_vars):
            for j in range(self.num_control_vars):
                if not fixed[i, j]:
                    parent_choice_rot_angle = np.random.randint(0, self.num_parents)
                    child_rot_angle[i, j] = parents[parent_choice_rot_angle][2][i, j]
                    child_rot_angle[j, i] = parents[parent_choice_rot_angle][2][j, i]
                    fixed[i, j] = True
                    fixed[j, i] = True

        return np.array(child_control_vars), np.array(child_stds), child_rot_angle

    def strategy_params_recombination(self, parent_strat_params1, parent_strat_params2):
        """

        :param parent_strategy_params: strategy params of 2 randomly sampled parents
        :return:
        """
        # Intermediate recombination of stds and rotation angles
        child_stds = self.recombination_weight * parent_strat_params1[0] + \
                     (1-self.recombination_weight) * parent_strat_params2[0]
        child_rot_angles = self.recombination_weight * parent_strat_params1[1] + \
                           (1-self.recombination_weight) * parent_strat_params2[1]

        return child_stds, child_rot_angles

    def recombination(self, parents):
        children = []
        for i in range(self.num_children):
            if self.global_recombination:
                # Global discrete recombination of all solution components
                child_control_vars, child_stds, child_rot_angles = self.global_recombinator(parents)
                children.append([child_control_vars, child_stds, child_rot_angles])
            else:
                # Randomly sample 2 parents
                parent_1 = 0
                parent_2 = 0
                while parent_1 == parent_2:
                    (parent_1, parent_2) = np.random.randint(0, self.num_parents, size=2)

                # Discrete recombination of control variables
                child_control_vars = self.control_var_recombination(parents[parent_1][0],
                                                                    parents[parent_2][0])
                # Intermediate recombination of strategy params
                child_stds, child_rot_angles = self.strategy_params_recombination(
                    parents[parent_1][1:], parents[parent_2][1:])
                children.append([child_control_vars, child_stds, child_rot_angles])

        return children

    def optimise(self):
        """

        :return:
        """
        children = self.generate_intial_population()
        # Store all control variable settings for each generation in history list
        children_control_vars_history = [[child[0] for child in children]]
        # Store all objective function values for each generation in history list
        children_fvals_history = []
        previous_parents = None
        total_func_evals = 0
        generation_times = []
        start = time.time()

        while total_func_evals < self.allowed_evals:
            # Assess population and selection parents
            parents, num_func_evals, children_fvals = self.select_parents(children,
                                                                          previous_parents)
            total_func_evals += num_func_evals
            children_fvals_history.append(children_fvals)

            # Mutate parents
            mutated_parents = self.mutate_solutions(parents)

            # Recombination
            children = self.recombination(mutated_parents)
            previous_parents = mutated_parents.copy()
            children_control_vars_history.append([child[0] for child in children])

            # Recording times
            generation_times.append(time.time() - start)

        # Final assessment outside the loop (can change code so don't have to do this?)
        final_children_fvals = list(map(self.objective_func, [soln[0] for soln in children]))
        children_fvals_history.append(final_children_fvals)
        generation_times.append(time.time() - start)

        return children_control_vars_history, children_fvals_history, generation_times

#
# test = EvolutionStrategy(schwefel_func, 5)
# pop = test.generate_intial_population()
# parents, _, _ = test.select_parents(pop, None)
# test.global_recombination(parents)

# # parents = test.mutate_solutions(parents)
# # new_pop = test.recombination(parents)
#
# control_vars_history, fvals_history = test.optimise()
# print(control_vars_history)
# print(len(control_vars_history))
# print(len(control_vars_history[0]))
# print(fvals_history)
# print(len(fvals_history))
# print(len(fvals_history[0]))

#
# cov_matrix = np.array([[3, 4, 7], [4, 3, -1], [7, -1, 8]])
# print(test.cov_matrix_to_rotation_angles(cov_matrix))