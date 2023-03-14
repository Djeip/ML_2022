import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

n = 100

k = 4

S = np.concatenate([np.array([None]), np.random.randint(1, 20, n)])

model = pyo.ConcreteModel()

model.S = S

model.M = pyo.RangeSet(k)
model.N = pyo.RangeSet(n)

model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

model.tmax = pyo.Var()
model.tmin = pyo.Var()


def obj_func(model):
    return model.tmax - model.tmin


model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

model.const2 = pyo.ConstraintList()

for i in model.N:
    model.const2.add(sum(model.x[i, j] for j in model.M) == 1)

for i in model.N:
    model.const2.add(sum(model.S[i] * model.x[i, j] for j in model.M) <= model.tmax)
    model.const2.add(sum(model.S[i] * model.x[i, j] for j in model.M) >= model.tmin)


def rule_const3(model):
    return model.tmin >= 0


def rule_const4(model):
    return model.tmax >= 0


model.const3 = pyo.Constraint(rule=rule_const3)

model.const4 = pyo.Constraint(rule=rule_const4)

model.pprint()

solver = SolverFactory("glpk", executable=r'C:\Users\Lenovo\w64\glpsol.exe')
result = solver.solve(model)

# Prints the results
print(result)

lst = list(model.x.keys())
list(model.x.get_values())

mat = np.split(np.array([model.x[a]() == 1 for a in lst]), k)

s1 = S[1:][mat[0]]
s2 = S[1:][mat[1]]
s3 = S[1:][mat[2]]
s4 = S[1:][mat[3]]

print(sum(s1), sum(s2), sum(s3), sum(s4))
