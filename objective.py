import numpy as np


def schwefel_func(x):
    x = np.array(x)
    return np.dot(-x, np.sin(np.sqrt(np.absolute(x))))
