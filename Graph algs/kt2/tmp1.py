import random

import numpy as np
import pandas as pd

data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\ML_2022\Graph algs\kt2\data.csv')
random.seed(50)
k = 100
S = data.values.T.tolist()[0]

S.sort()
S = S[::-1]


def normalize(probs):
    d = [1 / (1 + p) for p in probs]
    return np.array(list(map(lambda x: x / sum(d), d)))


def mask(S, k):
    n = len(S)
    matrix = np.zeros((n, k))
    for i in range(k):
        matrix[i, i] = 1
    for j in range(len(S)):
        prob = S @ matrix
        prob = normalize(prob)
        m = np.random.choice(list(range(k)), size=1, p=prob)[0]
        matrix[j, m] = 1
    return matrix


res = mask(S, k)


def main_metric(S, res):
    mn = np.mean(S)
    return sum([(t - mn) ** 2 for t in res])


main_metric(S, S @ res)
