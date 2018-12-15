import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    ax.plot_surface(xx, yy, z, cmap=cm.coolwarm)
    plt.show()


def contour_plot(objective_func):
    x = np.arange(-500, 500, 1)
    y = np.arange(-500, 500, 1)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = objective_func([xx[i, j], yy[i, j]])

    # plt.figure()
    plt.contour(xx, yy, z, 10)
    plt.show()


def search_pattern_contour_plot(objective_func, solns):
    x = np.arange(-500, 500, 1)
    y = np.arange(-500, 500, 1)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = objective_func([xx[i, j], yy[i, j]])

    x1s = [coord[0] for coord in solns]
    x2s = [coord[1] for coord in solns]

    plt.scatter(x1s, x2s, facecolors='none', edgecolors='r')
    plt.contour(xx, yy, z, 10)

    plt.show()


def fvals_iters_plot(fvals):
    plt.plot(fvals)
    plt.show()

def fvals_times_plot(fvals, times):
    plt.plot(times, fvals)
    plt.show()