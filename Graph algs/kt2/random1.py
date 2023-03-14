import random

import numpy as np
import pandas as pd

data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\ML_2022\Graph algs\kt2\data.csv')
random.seed(20)
k = 100
S = data.values.T.tolist()[0]


def mask(S, k):
    n = len(S)
    matrix = np.zeros((n, k))
    for i in range(k):
        matrix[i, i] = 1
    for j in range(len(S)):
        prob = S @ matrix
        m = pd.Series(prob).idxmin()
        matrix[j, m] = 1
    return matrix


res = mask(S, k)


def main_metric(S, res):
    mn = np.mean(S)
    return sum([(t - mn) ** 2 for t in res])


main_metric(S, S @ res)
