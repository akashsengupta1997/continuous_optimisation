import numpy as np
from annealer import Annealer
from evolution_strategy import EvolutionStrategy
from objective import schwefel_func


def optimiser_test(optimiser):
    """
    Function to test the best performance, average performance, consistency and computational
    runtime of given optimiser.
    :param optimiser: instance of Annealer class or EvolutionStrategy class
    :return: best objective found in 100 runs and corresponding solution, average optimal
    objective over 100 runs, standard deviation of optimal objectives over 100 runs, average
    runtime over 100 runs.
    """
    min_fvals = []
    min_solns = []
    run_times = []
    for i in range(100):
        np.random.seed(i)
        if isinstance(optimiser, Annealer):
            min_soln, min_fval, _, _, times = optimiser.simulated_annealing()
        elif isinstance(optimiser, EvolutionStrategy):
            min_soln, min_fval, _, _, times = optimiser.optimise()
        min_fvals.append(min_fval)
        min_solns.append(min_soln)
        run_times.append(times[-1])

    best_fval = min(min_fvals)  # best objective value in all 100 runs
    avg_fval = np.mean(min_fvals)  # average performance across the 100 runs
    best_soln = min_solns[min_fvals.index(best_fval)]  # best solution in all 100 runs
    std_fval = np.std(min_fvals)  # consistency across the 100 runs
    avg_runtime = np.mean(run_times)

    return best_fval, best_soln, avg_fval, std_fval, avg_runtime


# simple_annealer = Annealer(schwefel_func, 5)
# adaptive_soln_annealer = Annealer(schwefel_func, 5, adaptive_solns=True)
# adaptive_temp_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True)
# double_adaptive_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True,
#                                     adaptive_solns=True)
#
# print(optimiser_test(simple_annealer))
# print(optimiser_test(adaptive_soln_annealer))
# print(optimiser_test(adaptive_temp_annealer))
# print(optimiser_test(double_adaptive_annealer))

es = EvolutionStrategy(schwefel_func, 5)
es_elitist = EvolutionStrategy(schwefel_func, 5, elitist=True)
es_discrete = EvolutionStrategy(schwefel_func, 5, full_discrete_recombination=True)
es_global = EvolutionStrategy(schwefel_func, 5, global_recombination=True)

print(optimiser_test(es))
print(optimiser_test(es_elitist))
print(optimiser_test(es_discrete))
print(optimiser_test(es_global))
