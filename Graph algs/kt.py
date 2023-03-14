import numpy as np
from scipy.spatial import distance_matrix

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

n = 20
coordinates = np.random.random((n, 2))

dst_mtr = distance_matrix(coordinates, coordinates)

model = pyo.ConcreteModel()

model.M = pyo.RangeSet(n)
model.N = pyo.RangeSet(n)

model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: dst_mtr[i - 1][j - 1])

model.S = pyo.SetOf([1, 2, 3])
model.T = pyo.SetOf([9, 8])


def obj_func(model):
    return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)


model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)


def rule_const1(model, M):
    return sum(model.x[i, M] for i in model.N if i != M and i not in model.S and i not in model.T) == 1


model.const1 = pyo.Constraint(model.M, rule=rule_const1)


def rule_const2(model, N):
    return sum(model.x[N, j] for j in model.M if j != N and j not in model.S and j not in model.T) == 1


model.rest2 = pyo.Constraint(model.N, rule=rule_const2)


def rule_const3(model, S):
    return sum(model.x[S, j] for j in model.N if j != S and j not in model.T) == 1


model.const3 = pyo.Constraint(model.S, rule=rule_const3)


def rule_const4(model, T):
    return sum(model.x[i, T] for i in model.N if i != T and i not in model.S) == 1


model.const4 = pyo.Constraint(model.T, rule=rule_const4)

import itertools


def findsubsets(s, n):
    return [set(i) for j in range(2, n + 1) for i in itertools.combinations(s, j)]


model.sec = pyo.ConstraintList()

for sbst in findsubsets(model.M, 5):
    model.sec.add(sum(model.x[i, j] + model.x[j, i] for i, j in itertools.combinations(sbst, 2)) <= len(sbst) - 1)

model.pprint()

solver = SolverFactory("glpk", executable=r'C:\Users\Lenovo\w64\glpsol.exe')
result = solver.solve(model)

# Prints the results
print(result)

List = list(model.x.keys())
for i in List:
    if model.x[i]() == 1:
        print(i, '--', model.x[i]())
