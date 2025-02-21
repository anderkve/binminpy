import numpy as np
from binopy.BinnedOptimizer import BinnedOptimizer
from time import sleep

#
# User target function
#

def target_function(x, a):

    result = a + np.sum(x**2) 

    # sleep(0.5)
    # Waste some time in way that keeps the CPU working
    for i in range(int(1e7)):
        pass

    return result


#
# Test setup
#

binning_tuples = [(-3,3,1), (-3,3,1)]  # Currently not used

optimizer_kwargs = {
    "method": "L-BFGS-B",
    "tol": 1e-10,
    "args": (10),
}

max_processes = 4

binned_opt = BinnedOptimizer(
    target_function, 
    binning_tuples, 
    optimizer_kwargs=optimizer_kwargs, 
    max_processes=max_processes,
    return_evaluations=True,
)

collected_outputs = binned_opt.run()

print()

n_bins = len(collected_outputs)
for i in range(n_bins):
    print()
    print(f"Result bin {i}:")
    print(f"- optimization result:\n {collected_outputs[i][0]}")
    print(f"- x vals:\n {collected_outputs[i][1]}")
    print(f"- y vals:\n {collected_outputs[i][2]}")

print()


