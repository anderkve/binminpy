import numpy as np
from binminpy.BinnedOptimizer import BinnedOptimizer
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

output = binned_opt.run()

print()

n_bins = len(output["all_results"])
for i in range(n_bins):
    print()
    print(f"# Result bin {i}:")
    print(f"- optimization result:\n {output['all_results'][i][0]}")
    print(f"- x vals:\n {output['all_results'][i][1]}")
    print(f"- y vals:\n {output['all_results'][i][2]}")

print()
print()

best_bins = output["opt_bins"]
print(f"# Global optima found in bin(s) {best_bins}:")
for i,bin_index in enumerate(best_bins):
    print(f"- Bin {bin_index}:")
    print(f"  - x: {output['x_opt'][i]}")
    print(f"  - y: {output['y_opt'][i]}")

print()





