import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, OptimizeResult
from copy import copy
import warnings


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer_kwargs={}, max_processes=1, return_evaluations=False):
        """Constructor"""

        self.target_function = target_function
        self.binning_tuples = binning_tuples  # A list on the form [(x1_min, x1_max, n_bins_x1), (x2_min, x2_max, n_bins_x2), ...]
        self.optimizer_kwargs = optimizer_kwargs
        self.max_processes = max_processes
        self.return_evaluations = return_evaluations

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

        collected_outputs = [None] * n_bins

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:

            # Create a generator for all the tasks
            task_mapping = executor.map(self._worker_function, all_bounds)

            # Now execute the tasks in parallel and collect the results 
            for bin_id, result in enumerate(task_mapping):
                collected_outputs[bin_id] = result
                print(f"{self.print_prefix} Task {bin_id} is done.", flush=True)

        return collected_outputs

