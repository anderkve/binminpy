from binminpy.BinnedOptimizerMPIDynamic import BinnedOptimizerMPIDynamic
import numpy as np

# Define a simple quadratic target function.
def target_function(x):
    # Minimum at [1, 2]
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2


if __name__ == "__main__":

    # Define binning for two dimensions.
    # For example, first dimension from 0 to 2 (4 bins), second dimension from 0 to 4 (4 bins).
    binning_tuples = [(0, 2, 400), (0, 4, 400)]

    # Specify the optimizer and any keyword arguments.
    optimizer = "minimize"
    optimizer_kwargs = {"method": "L-BFGS-B"}

    # Create an instance of BinnedOptimizerMPIDynamic.
    bo = BinnedOptimizerMPIDynamic(
        target_function,
        binning_tuples,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        return_evals=True
    )

    # Run the optimization with task batches (e.g., 2 tasks per batch).
    result = bo.run(n_tasks_per_batch=100)

    # Only rank 0 (master) returns the full result.
    if result is not None:

        # print()
        # n_bins = len(result["all_optimizer_results"])
        # for i,bin_index_tuple in enumerate(result["bin_order"]):
        #     print()
        #     print(f"# Result for bin {bin_index_tuple} (all_optimizer_results[{i}]):")
        #     print(f"- bin limits: {bo.get_bin_limits(bin_index_tuple)}")
        #     print(f"- optimization result:\n {result['all_optimizer_results'][i]}")

        print()
        print()

        print("Optimization Result:")
        print("Optimal x:", result["x_optimal"])
        print("Optimal y:", result["y_optimal"])
        print("Optimal bins:", result["optimal_bins"])
