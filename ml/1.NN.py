import numpy as np

train = np.array([
    [[1, 0, 0], 0],
    [[0, 1, 0], 0],
    [[0, 0, 1], 1],
    [[1, 1, 0], 0],
    [[0, 1, 1], 1],
    [[1, 0, 1], 0],
    [[1, 1, 1], 0],
    [[0, 0, 0], 1],
])


def sig(x):
    return 1 / (1 + np.exp(-x))


O_1 = np.ones((2, 3))
O_2 = np.ones((1, 2))
n = 1
eps = 1e-10
maxiter = 10000

