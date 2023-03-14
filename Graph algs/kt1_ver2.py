import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

n = 10
alpha = 0.5
b3 = 3
b4 = 4
coordinates = np.random.random((n, 2))

dst_mtr = distance_matrix(coordinates, coordinates)

model = pyo.ConcreteModel()

model.M = pyo.RangeSet(n)
model.N = pyo.RangeSet(n)

model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: dst_mtr[i - 1][j - 1])


def reward(x):
    pass


def obj_func(model):
    return alpha * sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M) - (1 - alpha) * 1


model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)


def rule_const1(model, M):
    return sum(model.x[i, M] for i in model.N if i != M) == 1


model.const1 = pyo.Constraint(model.M, rule=rule_const1)


def rule_const2(model, N):
    return sum(model.x[N, j] for j in model.M if j != N) == 1


model.rest2 = pyo.Constraint(model.N, rule=rule_const2)

import itertools


def findsubsets(s, n):
    return [set(i) for j in range(2, n + 1) for i in itertools.combinations(s, j)]


model.sec = pyo.ConstraintList()

for sbst in findsubsets(model.M, 5):
    model.sec.add(sum(model.x[i, j] + model.x[j, i] for i, j in itertools.combinations(sbst, 2)) <= len(sbst) - 1)

model.pprint()

solver = SolverFactory("glpk", executable=r'C:\Users\Lenovo\w64\glpsol.exe')
result = solver.solve(model)

import pandas as pd

# Prints the results
print(result)


def func():
    res = []
    List = list(model.x.keys())
    for i in List:
        if model.x[i]() == 1:
            res.append(i)
            # print(i, '--', model.x[i]())

    way = []
    tmp = res[0]
    l = len(res)
    res.pop(0)

    while 2 * (l - 1) != len(way):
        for j in range(len(res)):
            if tmp[1] == res[j][0]:
                way += res[j]
                tmp = res[j]
                res.pop(j)
                break
    return


temp = pd.unique(way)
temp
P = [[1,2,3,4]]

def pairs(n):
    _temp = temp.tolist()
    _temp += _temp[:n]
    print(_temp)
    m = len(_temp) - n + 1
    return np.lib.stride_tricks.as_strided(_temp, shape=(m, n), strides=(4, 4))
