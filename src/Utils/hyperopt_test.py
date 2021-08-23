

import matplotlib
matplotlib.use("TkAgg")
from hyperopt import tpe, hp, fmin, rand
import matplotlib.pyplot as plt
import numpy as np


objective = lambda x: (x-3)**2 + 2


x = np.linspace(-10, 10, 100)
y = objective(x)

# fig = plt.figure()
# plt.plot(x, y)
# plt.show()



'''
The functions related to the search space are implemented in hyperopt.hp. The list is as follows.

hp.randint(label, upper) or hp.randint(label, low, high)
hp.uniform(label, low, high)
hp.loguniform(label, low, high)
hp.normal(label, mu, sigma)
hp.lognormal(label, mu, sigma)
hp.quniform(label, low, high, q)
hp.qloguniform(label, low, high, q)
hp.qnormal(label, mu, sigma, q)
hp.qlognormal(label, mu, sigma, q)
hp.choice(label, list)
hp.pchoice(label, p_list) with p_list as a list of (probability, option) pairs
hp.uniformint(label, low, high, q) or hp.uniformint(label, low, high) since q = 1.0

'''


# Define the search space of x between -10 and 10.
space = hp.uniform('x', -10, 10)
print(space)

best = fmin(
    fn=objective,     # Objective Function to optimize
    space=space,      # Hyperparameter's Search Space
    algo=atpe.suggest, # Optimization algorithm    # rand.sugget # atpe.suggest
    max_evals=1000    # Number of optimization attempts
)
print(best)




fig = plt.figure()
plt.plot(x, y)
plt.scatter(best['x'], objective(best['x']), color='red')
plt.show()