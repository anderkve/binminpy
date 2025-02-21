import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, OptimizeResult
from copy import copy
import warnings


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer_kwargs={}, max_processes=1, 
                 return_evaluations=False, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0):
        """Constructor"""

        self.target_function = target_function
        self.binning_tuples = binning_tuples  # A list on the form [(x1_min, x1_max, n_bins_x1), (x2_min, x2_max, n_bins_x2), ...]
        self.optimizer_kwargs = optimizer_kwargs
        self.max_processes = max_processes
        self.return_evaluations = return_evaluations
        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]

        self.print_prefix = "BinnedOptimizer:"

        if "bounds" in self.optimizer_kwargs:
            warnings.warn("BinnedOptimizer will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])


    def _worker_function(self, bounds):
        """Function to optimize the target function within a set of bounds"""

        use_optimizer_kwargs = copy(self.optimizer_kwargs)

        # Lists to store function evaluations
        x_points = []
        y_points = []

        # Wrapper for the target function, to allow us to save the evaluations
        def target_function_wrapper(x, *args):
            y = self.target_function(x, *args)
            print(f"{self.print_prefix} target_function_wrapper:  x: {x}  args: {args}  y: {y}")
            if self.return_evaluations:
                x_points.append(x)
                y_points.append(y)
            return y

        # Initial point for the optimization
        x0 = np.array([0.5 * (x_min + x_max) for x_min,x_max in bounds])

        # Do the optimization and store the result
        res = None
        try:
            res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)
        except ValueError as e:
            warnings.warn(f"{self.print_prefix} Optimization attempt returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
            use_optimizer_kwargs["method"] = "trust-constr"
            res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)

        if self.return_evaluations:
            return res, np.array(x_points), np.array(y_points)
        else:
            return res



    def run(self):
        """Start the optimization"""

        # Do the binning of the input space:
        # 
        # - Each bin needs
        #   1) a unique index
        #   2) a corresponding bounds list with the bin limits in each direction
        # 
        # - Save this in a list called "all_bounds", such that all_bounds[i]
        #   contains the bounds corresponding to bin i

        # Dummy example for a 2D function:
        all_bounds = [
            [(-2,2), (-2,2)], 
            [(-2,2), (2,4)], 
            [(-2,2), (4,6)],
            [(-2,2), (6,8)],

            [(-2,2), (-2,2)], 
            [(-2,2), (2,4)], 
            [(-2,2), (4,6)],
            [(-2,2), (6,8)],

            [(-2,2), (-2,2)], 
            [(-2,2), (2,4)], 
            [(-2,2), (4,6)],
            [(-2,2), (6,8)],

            [(-2,2), (-2,2)], 
            [(-2,2), (2,4)], 
            [(-2,2), (4,6)],
            [(-2,2), (6,8)],
        ]

        n_bins = len(all_bounds)

        output = {
            "x_opt": None,
            "y_opt": None,
            "opt_bins": None,
            "all_results": [None] * n_bins
        }
        

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:

            # Create a generator for all the tasks
            task_mapping = executor.map(self._worker_function, all_bounds)

            # Now execute the tasks in parallel and collect the results 
            for bin_index, result in enumerate(task_mapping):
                output["all_results"][bin_index] = result
                print(f"{self.print_prefix} Task {bin_index} is done.", flush=True)

        # Identify the global optima
        x_opt = []
        y_opt = [float('inf')]
        opt_bins = []
        for bin_index in range(n_bins):

            bin_result = None
            if self.return_evaluations:
                bin_result = output["all_results"][bin_index][0]
            else:
                bin_result = output["all_results"][bin_index]

            if bin_result is not None:
                if bin_result.fun < y_opt[0]:
                    x_opt = [bin_result.x]
                    y_opt = [bin_result.fun]
                    opt_bins = [bin_index]
                elif math.isclose(bin_result.fun, y_opt[0], rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                    x_opt.append(bin_result.x)
                    y_opt.append(bin_result.fun)
                    opt_bins.append(bin_index)

        output["x_opt"] = x_opt
        output["y_opt"] = y_opt
        output["opt_bins"] = opt_bins

        # We're done here
        return output

