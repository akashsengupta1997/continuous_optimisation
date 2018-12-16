import numpy as np
from annealer import Annealer
from objective import schwefel_func


def annealer_test(annealer):
    min_fvals = []
    min_solns = []
    for i in range(100):
        np.random.seed(i)
        min_soln, min_fval, _, _, _ = annealer.simulated_annealing()
        min_fvals.append(min_fval)
        min_solns.append(min_soln)

    best_fval = min(min_fvals)
    avg_fval = np.mean(min_fvals)
    best_soln = min_solns[min_fvals.index(best_fval)]
    std_fval = np.std(min_fvals)

    return best_fval, best_soln, avg_fval, std_fval


simple_annealer = Annealer(schwefel_func, 5)
adaptive_soln_annealer = Annealer(schwefel_func, 5, adaptive_solns=True)
adaptive_temp_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True)
double_adaptive_annealer = Annealer(schwefel_func, 5, adaptive_schedule=True,
                                     adaptive_solns=True)

print(annealer_test(simple_annealer))
print(annealer_test(adaptive_soln_annealer))
print(annealer_test(adaptive_temp_annealer))
print(annealer_test(double_adaptive_annealer))