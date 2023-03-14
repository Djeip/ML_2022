import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import itertools
import networkx as nx

n = 15

coordinates = np.random.random((n, 2))

dst_mtr = distance_matrix(coordinates, coordinates)

np.fill_diagonal(dst_mtr, 1e6)

model = pyo.ConcreteModel()

model.M = pyo.RangeSet(n)
model.N = pyo.RangeSet(n)

model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: dst_mtr[i - 1][j - 1])


def obj_func(model):
    return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)


model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)


def rule_const(model):
    return sum(model.x[i, j] for i in model.N for j in model.M) == n - 1


model.const = pyo.Constraint(rule=rule_const)


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

G = nx.from_numpy_matrix(dst_mtr)

nx.draw_networkx(G)
plt.show()
