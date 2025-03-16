import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, differential_evolution, basinhopping, shgo, dual_annealing, direct, OptimizeResult
from copy import copy
import warnings
import itertools

from binminpy.BinnedOptimizer import BinnedOptimizer

class BinnedOptimizerPPE(BinnedOptimizer):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, 
                         return_evals, optima_comparison_rtol, optima_comparison_atol)
        self.max_processes = max_processes


    def run(self):
        """Start the optimization"""

        output = {
            "x_optimal": None,
            "y_optimal": None,
            "optimal_bins": None,
            "bin_order": self.all_bin_index_tuples,
            "all_optimizer_results": [None] * self.n_bins,
            "x_evals": np.zeros((0, self.n_dims)),
            "y_evals": np.array([]),
        }
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:

            # Create a generator for all the tasks
            task_mapping = executor.map(self._worker_function, self.all_bin_index_tuples)

            # Now execute the tasks in parallel and collect the results 
            for bin_index, worker_output in enumerate(task_mapping):
                if self.return_evals:
                    opt_results, x_points, y_points = worker_output
                    output["all_optimizer_results"][bin_index] = opt_results
                    output["x_evals"] = np.vstack((output["x_evals"], x_points))
                    output["y_evals"] = np.hstack((output["y_evals"], y_points))
                else:
                    output["all_optimizer_results"][bin_index] = worker_output
                print(f"{self.print_prefix} Task {bin_index} is done.", flush=True)

        # Identify the global optima
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

        # We're done here
        return output

