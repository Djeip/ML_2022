import numpy as np
import random as rd

n = 100

k = 6

S = np.random.randint(1, 100, n)

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


mm,res = partition(S, 6)

T_mean = round(sum(S) / k)


def nij(t):
    return max(0, T_mean - t)








def ant(S,citys):
    first_number = rd.randint(0,citys-1)
    visited = [first_city]
    path =