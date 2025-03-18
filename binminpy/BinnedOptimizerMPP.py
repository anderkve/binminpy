import math
import numpy as np
from multiprocessing import Pool
from copy import copy
import warnings
import itertools

from binminpy.BinnedOptimizer import BinnedOptimizer

class BinnedOptimizerMPP(BinnedOptimizer):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, return_bin_centers=True, optima_comparison_rtol=1e-9, 
                 optima_comparison_atol=0.0, bin_masking=None):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                         return_bin_centers, optima_comparison_rtol, optima_comparison_atol, bin_masking)
        self.max_processes = max_processes


    def run(self):
        """Start the optimization using multiprocessing.Pool for parallel execution"""

        output = {
            "x_optimal": None,
            "y_optimal": None,
            "optimal_bins": None,
            "bin_tuples": np.array(self.all_bin_index_tuples, dtype=int),
            "x_optimal_per_bin": np.full((self.n_bins, self.n_dims), np.nan),
            "y_optimal_per_bin": np.full((self.n_bins,), np.inf),
            "all_optimizer_results": [None] * self.n_bins,
            "x_evals": None,
            "y_evals": None,
        }
        
        x_evals_list = []
        y_evals_list = []

        # Apply bin masking.
        use_bin_indices, use_bin_index_tuples = self._do_bin_masking()
        n_tasks = len(use_bin_indices)
        print(f"{self.print_prefix} The input space is binned using {self.n_bins} bins.", flush=True)
        print(f"{self.print_prefix} After applying the bin mask we are left with {n_tasks} optimization tasks.", flush=True)

        # Use multiprocessing Pool for parallel execution.
        with Pool(processes=self.max_processes) as pool:
            # pool.map returns a list of results
            task_results = pool.map(self._worker_function, use_bin_index_tuples)

            # Collect the results.
            for task_index, worker_output in enumerate(task_results):
                bin_index = use_bin_indices[task_index]
                bin_index_tuple = use_bin_index_tuples[task_index]
                task_number = task_index + 1

                opt_result, x_points, y_points = worker_output
                output["all_optimizer_results"][bin_index] = opt_result
                output["x_optimal_per_bin"][bin_index] = opt_result.x
                output["y_optimal_per_bin"][bin_index] = opt_result.fun
                if self.return_evals:
                    x_evals_list.extend(x_points)
                    y_evals_list.extend(y_points)
                print(f"{self.print_prefix} Task {task_number} with bin index tuple {bin_index_tuple} is done.", flush=True)

        if self.return_evals:
            output["x_evals"] = np.array(x_evals_list)
            output["y_evals"] = np.array(y_evals_list)

        # Identify the global optima.
        x_opt = []
        y_opt = [float('inf')]
        optimal_bins = []
        for bin_index in range(self.n_bins):
            bin_opt_result = output["all_optimizer_results"][bin_index]
            if bin_opt_result is not None:
                if bin_opt_result.fun < y_opt[0]:
                    x_opt = [bin_opt_result.x]
                    y_opt = [bin_opt_result.fun]
                    optimal_bins = [self.all_bin_index_tuples[bin_index]]
                elif math.isclose(bin_opt_result.fun, y_opt[0], rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                    x_opt.append(bin_opt_result.x)
                    y_opt.append(bin_opt_result.fun)
                    optimal_bins.append(self.all_bin_index_tuples[bin_index])
        output["x_optimal"] = x_opt
        output["y_optimal"] = y_opt
        output["optimal_bins"] = optimal_bins

        if self.return_bin_centers:
            output["bin_centers"] = np.empty((self.n_bins, self.n_dims), dtype=float)
            for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                output["bin_centers"][i] = self.get_bin_center(bin_index_tuple)

        return output
