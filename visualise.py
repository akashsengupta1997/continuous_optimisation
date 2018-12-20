import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from objective import schwefel_func

# TODO LABELS


def surface_plot(objective_func):
    x = np.arange(-500, 500, 0.5)
    y = np.arange(-500, 500, 0.5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = objective_func([xx[i, j], yy[i, j]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, label='2D-SF')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.show()


def contour_plot(objective_func):
    x = np.arange(-500, 500, 5)
    y = np.arange(-500, 500, 5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = objective_func([xx[i, j], yy[i, j]])

    # plt.figure()
    plt.contour(xx, yy, z, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def search_pattern_contour_plot(objective_func, solns):
    x = np.arange(-500, 500, 5)
    y = np.arange(-500, 500, 5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = objective_func([xx[i, j], yy[i, j]])

    x1s = [coord[0] for coord in solns]
    x2s = [coord[1] for coord in solns]

    plt.scatter(x1s, x2s, facecolors='none', edgecolors='r')
    plt.contour(xx, yy, z, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()


def fvals_iters_plot(fvals):
    plt.plot(fvals)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.show()


def fvals_times_plot(fvals, times):
    plt.plot(times, fvals)
    plt.xlabel('Runtime (s)')
    plt.ylabel('Objective value')
    plt.show()


def evolution_strat_fvals_generations_plot(fvals_history):
    num_generations = len(fvals_history)
    avg_fvals = [np.mean(fvals) for fvals in fvals_history]
    min_fvals = [min(fvals) for fvals in fvals_history]
    plt.plot(avg_fvals)
    plt.plot(min_fvals)
    plt.show()


def evolution_strat_fvals_evaluations_plot(fvals_history, num_evals_per_generation):
    evals = np.arange(len(fvals_history)) * num_evals_per_generation
    avg_fvals = [np.mean(fvals) for fvals in fvals_history]
    min_fvals = [min(fvals) for fvals in fvals_history]
    plt.plot(evals, avg_fvals)
    plt.plot(evals, min_fvals)
    plt.show()


def evolution_strat_fvals_time_plot(fvals_history, times):
    avg_fvals = [np.mean(fvals) for fvals in fvals_history]
    min_fvals = [min(fvals) for fvals in fvals_history]
    plt.plot(times, avg_fvals)
    plt.plot(times, min_fvals)
    plt.show()


def evolution_strat_search_pattern_contour_plot(objective_func, children_history,
                                                plot_generations):
    for generation in plot_generations:
        x = np.arange(-500, 500, 5)
        y = np.arange(-500, 500, 5)
        xx, yy = np.meshgrid(x, y, sparse=False)
        z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j] = objective_func([xx[i, j], yy[i, j]])

        control_vars = children_history[generation]
        x1s = [coord[0] for coord in control_vars]
        x2s = [coord[1] for coord in control_vars]

        plt.contour(xx, yy, z, 10)
        plt.scatter(x1s, x2s, facecolors='none', edgecolors='r')

        plt.show()

