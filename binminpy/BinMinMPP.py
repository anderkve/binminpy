import math
import numpy as np
from multiprocessing import Pool
from functools import partial
from copy import copy
import warnings
import itertools

from binminpy.BinMin import BinMin

class BinMinMPP(BinMin):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, return_bin_results=True, return_bin_centers=True, optima_comparison_rtol=1e-9, 
                 optima_comparison_atol=0.0, n_restarts_per_bin=1, bin_masking=None):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                         return_bin_results, return_bin_centers, optima_comparison_rtol, optima_comparison_atol,
                         n_restarts_per_bin, bin_masking)
        self.max_processes = max_processes


    def run(self):
        """Start the optimization using multiprocessing.Pool for parallel execution"""

        self.init_all_bin_index_tuples()

        output = {
            "x_optimal": None,
            "y_optimal": None,
            "optimal_bins": None,
            "bin_tuples": np.array(self.all_bin_index_tuples, dtype=int) if self.return_bin_results else None,
            "x_optimal_per_bin": np.full((self.n_bins, self.n_dims), np.nan) if self.return_bin_results else None,
            "y_optimal_per_bin": np.full((self.n_bins,), np.inf) if self.return_bin_results else None,
            "all_bin_results": [None] * self.n_bins if self.return_bin_results else None,
            "n_target_calls": 0,
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
            # Use functools.partial to fix a keyword argument in self._worker_function
            worker_function_partial = partial(self._worker_function, return_evals=self.return_evals)
            # pool.map returns a list of results
            task_results = pool.map(worker_function_partial, use_bin_index_tuples)

            # Collect the results.
            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for task_index, worker_output in enumerate(task_results):
                bin_index = use_bin_indices[task_index]
                bin_index_tuple = use_bin_index_tuples[task_index]
                task_number = task_index + 1
                opt_result, n_target_calls, x_points, y_points = worker_output
                # Update global optima?
                if (opt_result.fun < np.min(y_opt)) and (not math.isclose(opt_result.fun, np.min(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol)):
                    x_opt = [opt_result.x]
                    y_opt = [opt_result.fun]
                    optimal_bins = [bin_index_tuple]
                elif math.isclose(opt_result.fun, np.mean(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                    x_opt.append(opt_result.x)
                    y_opt.append(opt_result.fun)
                    optimal_bins.append(self.all_bin_index_tuples[bin_index])
                # Store some results
                if self.return_bin_results:
                    output["all_bin_results"][bin_index] = opt_result
                    output["x_optimal_per_bin"][bin_index] = opt_result.x
                    output["y_optimal_per_bin"][bin_index] = opt_result.fun
                output["n_target_calls"] += n_target_calls
                if self.return_evals:
                    x_evals_list.extend(x_points)
                    y_evals_list.extend(y_points)
                print(f"{self.print_prefix} Task {task_number} with bin index tuple {bin_index_tuple} is done.", flush=True)

        if self.return_evals:
            output["x_evals"] = np.array(x_evals_list)
            output["y_evals"] = np.array(y_evals_list)

        output["x_optimal"] = x_opt
        output["y_optimal"] = y_opt
        output["optimal_bins"] = optimal_bins

        if self.return_bin_centers:
            output["bin_centers"] = np.empty((self.n_bins, self.n_dims), dtype=float)
            for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                output["bin_centers"][i] = self.get_bin_center(bin_index_tuple)

        # We're done here
        return output
