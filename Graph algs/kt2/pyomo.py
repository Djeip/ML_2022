import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\ML_2022\Graph algs\kt2\data.csv')

k = 100

S = data.values.T.tolist()[0]

model = pyo.ConcreteModel()

model.S = S

model.M = pyo.RangeSet(len(S))
model.N = pyo.RangeSet(k)

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

mat = np.split(np.array([int(model.x[a]() == 1) for a in lst]), k)

results = [S @ x for x in mat]


# metric result

def main_metric(S, res):
    mn = np.mean(S)
    return sum([(t - mn) ** 2 for t in res])


main_metric(S, results)  # 29903143.512518007
