import numpy as np
import pandas as pd

results = np.random.uniform(1, 100, 30)
results.sort()
roulette = np.array(results).cumsum()/np.array(results).sum()

r = np.random.uniform(min(roulette), max(roulette))

print(pd.DataFrame(roulette).loc[roulette < r].index)
