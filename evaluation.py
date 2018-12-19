import numpy as np
from annealer import Annealer
from evolution_strategy import EvolutionStrategy
from objective import schwefel_func


def optimiser_test(optimiser):
    min_fvals = []
    min_solns = []
    for i in range(100):
        np.random.seed(i)
        if isinstance(optimiser, Annealer):
            min_soln, min_fval, _, _, _ = optimiser.simulated_annealing()
        elif isinstance(optimiser, EvolutionStrategy):
            min_soln, min_fval, _, _, _ = optimiser.optimise()
        min_fvals.append(min_fval)
        min_solns.append(min_soln)

    best_fval = min(min_fvals)  # best objective value in all 100 runs
    avg_fval = np.mean(min_fvals)  # average performance across the 100 runs
    best_soln = min_solns[min_fvals.index(best_fval)]  # best solution in all 100 runs
    std_fval = np.std(min_fvals)  # consistency across the 100 runs

    return best_fval, best_soln, avg_fval, std_fval


simple_annealer = Annealer(schwefel_func, 5)
adaptive_soln_annealer = Annealer(schwefel_func, 5, adaptive_solns=True)
adaptive_temp_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True)
double_adaptive_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True,
                                     adaptive_solns=True)

print(optimiser_test(simple_annealer))
print(optimiser_test(adaptive_soln_annealer))
print(optimiser_test(adaptive_temp_annealer))
print(optimiser_test(double_adaptive_annealer))

# es = EvolutionStrategy(schwefel_func, 5)
# es_elitist = EvolutionStrategy(schwefel_func, 5, elitist=True)
# es_global = EvolutionStrategy(schwefel_func, 5, global_recombination=True)
#
# print(optimiser_test(es))
# print(optimiser_test(es_elitist))
# print(optimiser_test(es_global))
