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

        self.num_parents = 1
        self.num_children = self.num_parents * 7

    def generate_intial_population(self):
        population = []
        for i in range(self.num_children):
            control_vars = 500*np.random.uniform(-1.0, 1.0, self.num_control_vars)
            stds = np.random.randn(self.num_control_vars)**2
            cov_matrix = np.diag(variances)
            population.append([control_vars, cov_matrix])

        return population

    def select_parents(self, children, previous_parents):
        if self.elitist:
            population = children + previous_parents
        else:
            population = children
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

    def rotation_angles_to_cov_matrix(self, rotation_angles, stds):
        """
        Diagonals are incorrect - actual diagonals are stds.
        :param rotation_angles:
        :return:
        """
        cov_matrix = np.zeros((self.num_control_vars, self.num_control_vars))
        for i in range(self.num_control_vars):
            for j in range(self.num_control_vars):
                if i != j:
                    cov_matrix[i, j] = 0.5*(stds[i]**2 - stds[j]**2)*np.tan(2*rotation_angles[i, j])

        return cov_matrix

    def mutate_stratetgy_params(self, population):
        tau = 1/math.sqrt(2*math.sqrt(self.num_control_vars))
        tau_prime = 1/math.sqrt(2*self.num_control_vars)
        beta = 0.0873
        for i in range(len(population)):
            chi_0 = np.random.randn()
            chi_i = np.random.randn(self.num_control_vars)
            chi_ij = np.random.randn(self.num_control_vars, self.num_control_vars)

            cov_matrix = population[i][1]
            stds = np.sqrt(np.diag(cov_matrix))
            new_stds = np.multiply(stds, np.exp(tau_prime * chi_0 + tau * chi_i))

            # For rotation angle matrices, only off-diagonal terms are relevant
            rot_angles = self.cov_matrix_to_rotation_angles(cov_matrix)
            print(cov_matrix)
            print(rot_angles)
            new_rot_angles = rot_angles + beta * chi_ij
            print(new_rot_angles)
            new_cov_matrix = self.rotation_angles_to_cov_matrix(new_rot_angles, new_stds) + np.diag(new_stds**2)
            # print(np.all(np.linalg.eigvals(cov_matrix) > 0))
            # print(np.all(np.linalg.eigvals(new_cov_matrix) > 0))
            # print(new_cov_matrix)
            # print(np.linalg.eigvals(new_cov_matrix))



    def mutate_control_variables(self, population):
        pass

    def recombine(self):
        pass


test = EvolutionStrategy(schwefel_func, 3)
pop = test.generate_intial_population()
# # print(test.select_parents(pop, None))
# test.mutate_stratetgy_params(pop)

cov_matrix = np.array([[3, 4, 7], [4, 3, -1], [7, -1, 8]])
print(test.cov_matrix_to_rotation_angles(cov_matrix))