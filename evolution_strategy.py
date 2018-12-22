import numpy as np
import math
import time


class EvolutionStrategy:
    """
    Contains all methods for evolutionary strategy optimisation.
    """
    def __init__(self, objective_func, num_control_vars, elitist=False,
                 full_discrete_recombination=False, global_recombination=False):
        """

        :param objective_func: function to minimise
        :param num_control_vars: number of control variables
        :param elitist: bool, use elitist selection if true
        :param full_discrete_recombination: bool, use local-discrete recombination on both
        control variables and strategy parameters if true.
        :param global_recombination: bool, use global-discrete recombination on both control
        variables and strategy parameters if true.
        """
        self.objective_func = objective_func
        self.num_control_vars = num_control_vars
        self.allowed_evals = 2000  # Don't need all 10000 allowed evaluations

        self.elitist = elitist
        self.full_discrete_recombination = full_discrete_recombination
        self.global_recombination = global_recombination

        self.num_parents = 55
        self.num_children = self.num_parents * 7
        self.recombination_weight = 0.5  # weight for intermediate recombination

    def generate_intial_population(self):
        """
        Generate an initial population with num_children solutions.

        Solutions are represented as a tuple with 3 elements. The first element is a
        5-dimensional vector representing the control variables. The second element is
        a 5-dimensional vector representing variances/step sizes. The third element,
        representing 10 rotation angles, is a 5-by-5 skew-symmetric matrix
        with all diagonal elements equal to 0.

        The population is then a list of tuples.

        :return: initial population.
        """
        population = []
        for i in range(self.num_children):
            control_vars = 500*np.random.uniform(-1.0, 1.0, self.num_control_vars)
            # Choose small initial values to get PSD covariance matrix post-mutation
            stds = 0.01*np.ones(self.num_control_vars)
            rot_angles = np.zeros((self.num_control_vars, self.num_control_vars))
            population.append([control_vars, stds, rot_angles])

        return population

    def select_parents(self, children, previous_parents):
        """
        Deterministically select parents from children or from children + previous parents,
        depending on whether self.elitist is true or not.
        :param children: list of solutions (tuples)
        :param previous_parents: list of solutions (tuples)
        :return: parents
        """
        if self.elitist and previous_parents is not None:
            population = children + previous_parents
        else:
            population = children

        # Check that all control variables within bounds - store the population indices of
        # invalid solutions, for later removal from population
        invalid_indices = []
        for i in range(len(population)):
            control_vars = population[i][0]
            if np.any(control_vars > 500) or np.any(control_vars < -500):
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
        Construct a PSD covariance matrix from rotation angles and standard deviations (for a
        single solution).
        :param rotation_angles
        :param stds
        :return: covariance matrix
        """
        cov_matrix = np.zeros((self.num_control_vars, self.num_control_vars))
        for i in range(self.num_control_vars):
            for j in range(self.num_control_vars):
                if i == j:
                    cov_matrix[i, j] = stds[i] ** 2
                else:
                    cov_matrix[i, j] = 0.5 * (stds[i] ** 2 - stds[j] ** 2) * np.tan(
                        2 * rotation_angles[i, j])

        # Ensure that covariance matrix is positive definite by adding eI till all
        # eigenvalues > 0
        i = 0
        epsilon = 0.1
        while not np.all(np.linalg.eigvals(cov_matrix) > 0):
            cov_matrix = cov_matrix + epsilon*np.identity(self.num_control_vars)
            i = i + 1
            if i > 30:
                epsilon = 1
        return cov_matrix

    def mutate_stratetgy_params(self, solution):
        """
        Mutate strategy parameters using Eqns 3 and 4 from ES handout.
        :param solution: tuple, solution[1] and solution[2] are the stds and rot_angles to be
        mutated
        :return: mutated stds and rot_angles
        """
        tau = 1/math.sqrt(2*math.sqrt(self.num_control_vars))
        tau_prime = 1/math.sqrt(2*self.num_control_vars)
        beta = 0.0873

        chi_0 = np.random.randn()
        chi_i = np.random.randn(self.num_control_vars)
        temp = np.sqrt(2)*np.random.randn(self.num_control_vars, self.num_control_vars)
        chi_ij = (temp - temp.T)/2  # skew-symmetric
        # multiplying temp by sqrt(2) to make chi_ij ~ N(0,1)

        stds = solution[1]
        new_stds = np.multiply(stds, np.exp(tau_prime * chi_0 + tau * chi_i))

        # For rotation angle matrices, only off-diagonal terms are relevant
        rot_angles = solution[2]
        new_rot_angles = rot_angles + beta * chi_ij

        return new_stds, new_rot_angles

    def mutate_solutions(self, parents):
        """
        Mutate parents before recombination. First mutate strategy params, then mutate control
        variables (Eqns 3, 4, 5, 6 from ES handout).
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

    def control_var_discrete_recombination(self, parent_control_vars1, parent_control_vars2):
        """
        Discrete recombination of control variables.
        :param parent_control_vars1: control variables of 1 of the 2 randomly sampled parents
        i.e. list
        :param parent_control_vars2: control variables of 1 of the 2 randomly sampled parents
        i.e. list
        :return: child control variables (list)
        """
        # Discrete recombination
        cross_points = np.random.rand(self.num_control_vars) < 0.5  # p(cross) = 0.5 (fair coin toss)
        child_control_vars = np.where(cross_points, parent_control_vars1, parent_control_vars2)

        return child_control_vars

    def global_discrete_recombination(self, parents):
        """
        Global discrete recombination of control variables and strategy parameters.
        :param parents: All num_parents parent from this generation, i.e. list of tuples.
        :return: child solution (tuple)
        """
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

    def strategy_params_intermediate_recombination(self, parent_strat_params1,
                                                   parent_strat_params2):
        """
        Intermediate recombination of strategy parameters.
        :param parent_strat_params1: strategy params of 1 of 2 randomly sampled parents
        (tuple of list and matrix)
        :param parent_strat_params2: strategy params of 1 of 2 randomly sampled parents
        (tuple of list and matrix)
        :return: child strategy params (tuple of list and matrix)
        """
        # Intermediate recombination of stds and rotation angles
        child_stds = self.recombination_weight * parent_strat_params1[0] + \
                     (1-self.recombination_weight) * parent_strat_params2[0]
        child_rot_angles = self.recombination_weight * parent_strat_params1[1] + \
                           (1-self.recombination_weight) * parent_strat_params2[1]

        return child_stds, child_rot_angles

    def strategy_params_discrete_recombination(self, parent_strat_params1,
                                               parent_strat_params2):
        """
        Discrete recombination of strategy parameters.
        :param parent_strat_params1:strategy params of 1 of 2 randomly sampled parents
        (tuple of list and matrix)
        :param parent_strat_params2: strategy params of 1 of 2 randomly sampled parents
        (tuple of list and matrix)
        :return: child strategy params (tuple of list and matrix)
        """
        std_cross_points = np.random.rand(self.num_control_vars) < 0.5  # p(cross) = 0.5 (fair coin toss)
        child_stds = np.where(std_cross_points, parent_strat_params1[0], parent_strat_params2[0])

        temp = np.random.rand(self.num_control_vars, self.num_control_vars)
        temp = (temp + temp.T) / 2
        rot_angle_cross_points = temp < 0.5
        child_rot_angles = np.where(rot_angle_cross_points, parent_strat_params1[1],
                                    parent_strat_params2[1])
        return child_stds, child_rot_angles

    def recombination(self, parents):
        """
        Recombination between parents. Default recombination configuration is discrete
        recombination on control variables and intermediate recombination on strategy
        parameters.
        :param parents: list of tuples, each representing a single parent solution.
        :return: children, list of tuples, each representing a single solution,
        len = self.num_children
        """
        children = []
        for i in range(self.num_children):
            if self.global_recombination:
                # Global discrete recombination of all solution components
                child_control_vars, child_stds, child_rot_angles = self.global_discrete_recombination(parents)
                children.append([child_control_vars, child_stds, child_rot_angles])
            else:
                # Randomly sample 2 parents
                parent_1 = 0
                parent_2 = 0
                while parent_1 == parent_2:
                    (parent_1, parent_2) = np.random.randint(0, self.num_parents, size=2)

                # Discrete recombination of control variables
                child_control_vars = self.control_var_discrete_recombination(parents[parent_1][0],
                                                                             parents[parent_2][0])
                if self.full_discrete_recombination:
                    # Discrete recombination of strategy params
                    child_stds, child_rot_angles = self.strategy_params_discrete_recombination(
                        parents[parent_1][1:], parents[parent_2][1:])
                    children.append([child_control_vars, child_stds, child_rot_angles])
                else:
                    # Intermediate recombination of strategy params
                    child_stds, child_rot_angles = self.strategy_params_intermediate_recombination(
                        parents[parent_1][1:], parents[parent_2][1:])
                    children.append([child_control_vars, child_stds, child_rot_angles])

        return children

    def optimise(self):
        """
        Perform evolutionary strategy algorithm.
        :return: best setting of control variables, best function value, history of control
        variable solutions in each generation, history of fvals in each generation, computation
        time for each generation.
        """
        children = self.generate_intial_population()
        # Store all control variable settings for each generation in history list
        children_control_vars_history = [[child[0] for child in children]]
        # Store all objective function values for each generation in history list
        children_fvals_history = []
        previous_parents = None
        total_func_evals = 0
        best_fval = np.inf
        best_control_vars = None
        generation_times = []
        start = time.time()

        while total_func_evals < self.allowed_evals:
            # Assess population and select parents
            parents, num_func_evals, children_fvals = self.select_parents(children,
                                                                          previous_parents)
            total_func_evals += num_func_evals
            children_fvals_history.append(children_fvals)

            # If best child solution found in this generation, save it.
            # Note that children CAN have control variable values outside [-500, 500], although
            # it is ensured that these invalid children are not selected to be parents.
            # Need to check to ensure that best child solution is a valid solution.
            # parents list is sorted - parents[0] is best child solution in this generation.
            best_generation_fval = self.objective_func(parents[0][0])
            total_func_evals += 1
            if best_generation_fval < best_fval:
                if np.all(parents[0][0] < 500) \
                        and np.all(parents[0][0] > -500):
                    best_fval = best_generation_fval
                    best_control_vars = parents[0][0]

            # Mutate parents
            mutated_parents = self.mutate_solutions(parents)

            # Recombination
            children = self.recombination(mutated_parents)
            previous_parents = mutated_parents.copy()
            children_control_vars_history.append([child[0] for child in children])

            # Recording times
            generation_times.append(time.time() - start)

        # Final assessment outside the loop - note: not same structure as Fig 1 in ES handout
        final_parents, _, final_children_fvals = self.select_parents(children,
                                                                     previous_parents)
        children_fvals_history.append(final_children_fvals)
        best_final_fval = self.objective_func(final_parents[0][0])
        if best_final_fval < best_fval:
            if np.all(final_parents[0][0] < 500) \
                    and np.all(final_parents[0][0] > -500):
                best_fval = best_final_fval
                best_control_vars = final_parents[0][0]
        generation_times.append(time.time() - start)

        return best_control_vars, best_fval, \
               children_control_vars_history, children_fvals_history, generation_times

