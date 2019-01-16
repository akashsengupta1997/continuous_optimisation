import numpy as np


def schwefel_func(x):
    """
    Function to evaluate Schwefel's function at input x. Dimensionality depends on length of x.
    :param x: list/numpy array
    :return: f(x): float
    """
    x = np.array(x)
    return np.dot(-x, np.sin(np.sqrt(np.absolute(x))))
