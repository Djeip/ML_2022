import numpy as np
import pandas as pd


data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\ML_2022\Graph algs\kt2\data.csv')

k = 100

S = data.values.T.tolist()[0]


S.sort()
S = S[::-1]


def mask(k, n):
    matrix = np.zeros((n, k))
    for i in range(n):
        matrix[i][i % k] = 1
    return matrix


def partition(lst, k):
    msk = mask(k, len(lst))
    res = S @ msk
    print(f'sum in each part: {res}')
    print(f'F:{res.max() - res.min()}')
    return [i % k for i in range(len(lst))],res


a,b = partition(S,k)
