import numpy as np
from binopy.BinnedOptimizer import BinnedOptimizer


# User target function
def target_function(x):
    result = 10. + np.sum(x**2)
    return result


# Test setup
binning_tuples = [(-3,3,1), (-3,3,1)]

optimizer_kwargs = {
    "method": "L-BFGS-B",
    "tol": 1e-9,
}

binned_opt = BinnedOptimizer(target_function, binning_tuples, optimizer_kwargs=optimizer_kwargs, max_processes=1)


bounds = [(-3,3), (-3,3)]
res = binned_opt.worker_function(bounds)

print(res)

