import numpy as np
import math
import time
from objective import schwefel_func

# TODO put all methods together into final function - best soln, best fval and history of all POPULATIONS and fvals
# TODO count objective function evaluations in select_parents - might be good idea to return
# TODO implement global recombination
# TODO choose population size - computation vs accuracy tradeoff


class EvolutionStrategy:
    def __init__(self, objective_func, num_control_vars, elitist=False,
                 global_recombination=False):

        self.objective_func = objective_func
        self.num_control_vars = num_control_vars
        self.allowed_evals = 10000

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
        if self.elitist:
            population = children + previous_parents
        else:
            population = children

        # Check that all control variables within bounds - don't select solutions with invalid
        # control variables
        for solution in population:
            control_vars = solution[0]
            if np.any(control_vars > 500):
                population.remove(solution)
        fvals = list(map(self.objective_func, [solution[0] for solution in population]))
        sorted_indices = np.argsort(fvals)
        parents = []
        for i in range(self.num_parents):
            parents.append(children[sorted_indices[i]])

        return parents

    def cov_matrix_to_rotation_angles(self, covariance_matrix):
        rot_angles = np.zeros((self.num_control_vars, self.num_control_vars))
        for i in range(self.num_control_vars):
            for j in range(self.num_control_vars):
                if i != j:
                    # Only want to calculate for off-diagonal elements
                    # Need to check that var_i != var_j or else will get divide by 0 errors
                    # if var_i == var_j, arctan function should return pi/2 or -pi/2
                    # Edge case when c_ij = 0 and var_i = var_j: get 0/0 in arctan - in this
                    # case, I am just assigning alpha_ij = 0
                    if covariance_matrix[i, j] == 0:
                        rot_angles[i, j] = 0
                    elif covariance_matrix[i, i] == covariance_matrix[j, j]:
                        rot_angles[i, j] = 0.5*math.copysign(1, covariance_matrix[i, j])*np.pi/2
                    else:
                        rot_angles[i, j] = 0.5*np.arctan(2*covariance_matrix[i, j]/(covariance_matrix[i, i] - covariance_matrix[j, j]))

        return rot_angles

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
            # print(new_stds)
            # print(new_rot_angles)
            # Cov_matrix should be symmetric since rot_angles matrix is skew-symmetric
            cov_matrix = self.construct_cov_matrix(new_rot_angles, new_stds)

            # print(cov_matrix)
            # print(np.all(np.linalg.eigvals(cov_matrix) > 0))
            # print(np.linalg.eigvals(cov_matrix))
            n = np.random.multivariate_normal(np.zeros(self.num_control_vars), cov_matrix, check_valid='warn')
            new_control_vars = solution[0] + n
            mutated_population.append([new_control_vars, new_stds, new_rot_angles])
        return mutated_population

    def control_var_recombination(self, parent_1, parent_2):
        """

        :param parent_control_vars: control varaibles of 2 randomly sampled parents
        :return:
        """
        # Discrete recombination
        cross_points = np.random.rand(self.num_control_vars) < 0.5  # p(cross) = 0.5 (fair coin toss)
        child_control_vars = np.where(cross_points, parent_1, parent_2 )

        return child_control_vars

    def strategy_params_recombination(self, parent_1, parent_2):
        """

        :param parent_strategy_params: strategy params of 2 randomly sampled parents
        :return:
        """
        child_strategy_params = [None, None]
        # Intermediate recombination
        child_strategy_params[0] = self.recombination_weight*parent_1[0] + \
            (1-self.recombination_weight)*parent_2[0]
        child_strategy_params[1] = self.recombination_weight*parent_1[1] + \
            (1-self.recombination_weight)*parent_2[1]

        return child_strategy_params

    def recombination(self, parents):
        children = []
        for i in range(self.num_children):
            if self.global_recombination:
                pass
            else:
                # Randomly sample 2 parents
                parent_1 = 0
                parent_2 = 0
                while parent_1 == parent_2:
                    (parent_1, parent_2) = np.random.randint(0, self.num_parents, size=2)
                # Discrete recombination of control variables
                child_control_vars = self.control_var_recombination(parents[parent_1][0],
                                                                    parents[parent_2][0])
                child_strat_params = self.strategy_params_recombination(parents[parent_1][1:],
                                                                        parents[parent_2][1:])
                # print('parent 1', parents[parent_1])
                # print('parent 2', parents[parent_2])
                # print('child', [child_control_vars, child_strat_params])
                children.append([child_control_vars, child_strat_params])

        return children



test = EvolutionStrategy(schwefel_func, 5)
pop = test.generate_intial_population()
parents = test.select_parents(pop, None)
parents = test.mutate_solutions(parents)
new_pop = test.recombination(parents)

#
# cov_matrix = np.array([[3, 4, 7], [4, 3, -1], [7, -1, 8]])
# print(test.cov_matrix_to_rotation_angles(cov_matrix))