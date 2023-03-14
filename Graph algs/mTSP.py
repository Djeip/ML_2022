import numpy as np
from scipy.spatial import distance_matrix
import itertools
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

n = 10
coordinates = np.random.random((n, 2))

dst_mtr = distance_matrix(coordinates, coordinates)

m = 5  # salesmen cnt
p = n  # max nodes by salesmen

print(m * p >= n)

model = pyo.ConcreteModel()

model.M = pyo.RangeSet(n)
model.N = pyo.RangeSet(n)

model.x = pyo.Var(model.N, model.M, within=pyo.Binary)
model.u = pyo.Var(model.N, within=pyo.Reals)

model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: dst_mtr[i - 1][j - 1])


def obj_func(model):
    return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)


model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)


def rule_const0(model, N):
    return model.u[N] >= 0


model.const0 = pyo.Constraint(model.N, rule=rule_const0)


def rule_const1(model, M):
    return sum(model.x[i, M] for i in model.N if i != M) == 1


model.const1 = pyo.Constraint(model.M, rule=rule_const1)


def rule_const2(model, N):
    return sum(model.x[N, j] for j in model.M if j != N) == 1


model.const2 = pyo.Constraint(model.N, rule=rule_const2)


def rule_const3(model):
    return sum(model.x[i, 1] for i in model.N if i != 1) == m


model.const3 = pyo.Constraint(rule=rule_const3)


def rule_const4(model):
    return sum(model.x[1, j] for j in model.M if j != 1) == m


model.const4 = pyo.Constraint(rule=rule_const4)

model.const5 = pyo.ConstraintList()

for i, j in itertools.product(model.M, model.N):
    if i != j:
        model.const5.add(model.u[i] - model.u[j] + p * model.x[i, j] <= p - 1)

model.pprint()

solver = SolverFactory("glpk", executable=r'C:\Users\Lenovo\w64\glpsol.exe')
result = solver.solve(model)

# Prints the results
print(result)

List = list(model.x.keys())
for i in List:
    if model.x[i]() == 1:
        print(i, '--', model.x[i]())
