from binminpy.BinnedOptimizerMPI import BinnedOptimizerMPI
import numpy as np

# Define a simple target function.
def target_function(x):
    # A quadratic function with a minimum at [1, 2].
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

if __name__ == "__main__":
    # Define binning for two dimensions.
    # For example, the first dimension ranges from 0 to 2 split into 4 bins,
    # and the second from 0 to 4 split into 4 bins.
    binning_tuples = [(0, 2, 10), (0, 4, 10)]

    # Specify the optimizer and any kwargs.
    optimizer = "minimize"
    optimizer_kwargs = {"method": "L-BFGS-B"}

    # Instantiate the BinnedOptimizerMPI.
    bo = BinnedOptimizerMPI(
        target_function,
        binning_tuples,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        return_evals=True
    )

    # Run the optimization.
    result = bo.run(task_distribution="dynamic")

    # Only rank 0 will have the complete result.
    if result is not None:

        print()
        n_bins = len(result["all_optimizer_results"])
        for i,bin_index_tuple in enumerate(result["bin_order"]):
            print()
            print(f"# Result for bin {bin_index_tuple} (all_optimizer_results[{i}]):")
            print(f"- bin limits: {bo.get_bin_limits(bin_index_tuple)}")
            print(f"- optimization result:\n {result['all_optimizer_results'][i]}")

        print()
        print()

        print("Optimization Result:")
        print()
        print("x_evals:", result["x_evals"])
        print()
        print("Optimal x:", result["x_optimal"])
        print("Optimal y:", result["y_optimal"])
        print("Optimal bins:", result["optimal_bins"])
