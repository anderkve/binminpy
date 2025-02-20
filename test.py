import numpy as np
from binopy.BinnedOptimizer import BinnedOptimizer
from time import sleep


# User target function
def target_function(x):
    result = 10. + np.sum(x**2)
    sleep(0.5)
    print(f"x: {x}   target_function: {result}")
    return result


# Test setup
binning_tuples = [(-3,3,1), (-3,3,1)]

optimizer_kwargs = {
    "method": "L-BFGS-B",
    "tol": 1e-10,
}

max_processes = 4
binned_opt = BinnedOptimizer(target_function, binning_tuples, optimizer_kwargs=optimizer_kwargs, max_processes=max_processes)

collected_results = binned_opt.run()

for i,res in enumerate(collected_results):
    print(f"------ Bin {i} ------ ")
    print(res)
    print()



