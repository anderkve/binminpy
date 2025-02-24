import numpy as np
from binminpy.BinnedOptimizer import BinnedOptimizer
from time import sleep

#
# User target function
#

def target_function(x, a):

    result = a + np.sum(x**2) 
    # result = a + np.sum(np.sin(x**2)) 

    # sleep(0.5)
    # Waste some time in way that keeps the CPU working
    # for i in range(int(1e6)):
    #     pass

    return result


#
# Test setup
#

binning_tuples = [(-2.5, 2.5, 5)]*3


optimizer_kwargs = {
    "method": "L-BFGS-B",
    # "tol": 1e-6,
    "args": (10),
}

max_processes = 4

binned_opt = BinnedOptimizer(
    target_function, 
    binning_tuples,
    optimizer="minimize", 
    # optimizer="differential_evolution", 
    # optimizer="basinhopping", 
    # optimizer="shgo", 
    # optimizer="dual_annealing", 
    # optimizer="direct", 
    optimizer_kwargs=optimizer_kwargs, 
    max_processes=max_processes,
    return_evals=True,
)

output = binned_opt.run()

print()

n_bins = len(output["all_optimizer_results"])
for i,bin_index_tuple in enumerate(output["bin_order"]):
    print()
    print(f"# Result for bin {bin_index_tuple} (all_optimizer_results[{i}]):")
    print(f"- bin limits: {binned_opt.get_bin_limits(bin_index_tuple)}")
    print(f"- optimization result:\n {output['all_optimizer_results'][i]}")
    # print(f"- x vals:\n {output['all_evals'][i][1]}")
    # print(f"- y vals:\n {output['all_evals'][i][2]}")

print()
print()

best_bins = output["optimal_bins"]
print(f"# Global optima found in bin(s) {best_bins}:")
for i,bin_index_tuple in enumerate(best_bins):
    print(f"- Bin {bin_index_tuple}:")
    print(f"  - bin limits: {binned_opt.get_bin_limits(bin_index_tuple)}")
    print(f"  - x: {output['x_optimal'][i]}")
    print(f"  - y: {output['y_optimal'][i]}")

print()


# x_points, y_points = binned_opt.concatenate_evals(output["all_evals"])
x_points = output["x_evals"]
y_points = output["y_evals"]

print()
print(x_points)
print()
print(y_points)

# n_pts = len(y_points)
# print()
# for i in range(n_pts):
#     print(f"  x: {x_points[i]}   y: {y_points[i]}")
# print()
