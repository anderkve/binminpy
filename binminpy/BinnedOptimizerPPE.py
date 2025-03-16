import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, differential_evolution, basinhopping, shgo, dual_annealing, direct, OptimizeResult
from copy import copy
import warnings
import itertools


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0):
        """Constructor"""

        self.target_function = target_function
        self.binning_tuples = binning_tuples  # A list on the form [(x1_min, x1_max, n_bins_x1), (x2_min, x2_max, n_bins_x2), ...]
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.max_processes = max_processes
        self.return_evals = return_evals
        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]
        self.n_bins = np.prod([self.n_bins_per_dim])
        self.bin_limits_per_dim = [np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1) for d in range(self.n_dims)]
        self.all_bin_index_tuples = list(itertools.product(*[range(self.n_bins_per_dim[d]) for d in range(self.n_dims)]))

        self.print_prefix = "BinnedOptimizer:"

        known_optimizers = ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct"]
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers}.")

        if "bounds" in self.optimizer_kwargs:
            warnings.warn("BinnedOptimizer will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])

        # Ensure that self.optimizer_kwargs["args"] is a tuple 
        if "args" in self.optimizer_kwargs:
            if not isinstance(self.optimizer_kwargs["args"], tuple):
                self.optimizer_kwargs["args"] = tuple([self.optimizer_kwargs["args"]])


    def get_bin_limits(self, bin_index_tuple):
        bounds = []
        for d in range(self.n_dims):
            index_d = bin_index_tuple[d]
            # Add a tuple (x_d_min, x_d_max) for dimension d
            bounds.append((self.bin_limits_per_dim[d][index_d], self.bin_limits_per_dim[d][index_d+1]))
        return bounds


    def _worker_function(self, bin_index_tuple):
        """Function to optimize the target function within a set of bounds"""
        bounds = self.get_bin_limits(bin_index_tuple)
        use_optimizer_kwargs = copy(self.optimizer_kwargs)

        # Lists to store function evaluations
        x_points = []
        y_points = []

        # Wrapper for the target function, to allow us to save the evaluations
        def target_function_wrapper(x, *args):
            y = self.target_function(x, *args)
            if self.return_evals:
                x_points.append(x)
                y_points.append(y)
            return y

        # Initial point (for optimizers that need this)
        x0 = np.array([0.5 * (x_min + x_max) for x_min,x_max in bounds])

        # Do the optimization and store the result
        res = None

        if self.optimizer == "minimize":
            try:
                res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)
            except ValueError as e:
                warnings.warn(f"{self.print_prefix} scipy.optimize.minimize returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
                use_optimizer_kwargs["method"] = "trust-constr"
                res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)

        elif self.optimizer == "differential_evolution":
            res = differential_evolution(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "basinhopping":
            if not "minimizer_kwargs" in use_optimizer_kwargs:
                use_optimizer_kwargs["minimizer_kwargs"] = {}
            use_optimizer_kwargs["minimizer_kwargs"]["bounds"] = bounds
            if "args" in use_optimizer_kwargs:
                use_optimizer_kwargs["minimizer_kwargs"]["args"] = copy(use_optimizer_kwargs["args"])
                del(use_optimizer_kwargs["args"])
            res = basinhopping(target_function_wrapper, x0, **use_optimizer_kwargs)

        elif self.optimizer == "shgo":
            res = shgo(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "dual_annealing":
            res = dual_annealing(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "direct":
            res = direct(target_function_wrapper, bounds, **use_optimizer_kwargs)

        if self.return_evals:
            return res, np.array(x_points), np.array(y_points)
        else:
            return res


    def run(self):
        """Start the optimization"""

        output = {
            "x_optimal": None,
            "y_optimal": None,
            "optimal_bins": None,
            "bin_order": self.all_bin_index_tuples,
            "all_optimizer_results": [None] * self.n_bins,
            # "all_evals": [None] * self.n_bins,
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
                    # output["all_evals"][bin_index] = (x_points, y_points)
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

