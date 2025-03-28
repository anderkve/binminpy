import math
import numpy as np
from scipy.optimize import OptimizeResult
from copy import copy
import warnings
import itertools


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, 
                 return_evals=False, return_bin_centers=True, optima_comparison_rtol=1e-9, 
                 optima_comparison_atol=0.0, bin_masking=None):
        """Constructor.

        Parameters:
          target_function: function to optimize.
          binning_tuples: list of tuples [(min, max, n_bins), ...] for each dimension.
          optimizer: string, one of ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct", "iminuit", "diver"].
          optimizer_kwargs: additional keyword arguments for the optimizer.
          return_evals: if True, record evaluations.
          optima_comparison_rtol, optima_comparison_atol: tolerances for comparing optima.
          bin_masking: a function on the form bin_masking(bin_centre, bin_limits) -> True/False.
        """

        self.target_function = target_function
        self.binning_tuples = binning_tuples
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.return_evals = return_evals
        self.return_bin_centers = return_bin_centers
        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol
        self.bin_masking = bin_masking

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]
        self.n_bins = np.prod([self.n_bins_per_dim])
        self.bin_limits_per_dim = [np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1) for d in range(self.n_dims)]
        self.all_bin_index_tuples = list(itertools.product(*[range(self.n_bins_per_dim[d]) for d in range(self.n_dims)]))

        self.print_prefix = "BinnedOptimizer:"

        known_optimizers = ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct", "iminuit", "diver"]
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers}.")

        if "bounds" in self.optimizer_kwargs:
            if self.optimizer_kwargs["bounds"] is not None:
                warnings.warn("BinnedOptimizer will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])

        # Ensure that self.optimizer_kwargs["args"] is a tuple 
        if "args" in self.optimizer_kwargs:
            if not isinstance(self.optimizer_kwargs["args"], tuple):
                self.optimizer_kwargs["args"] = tuple([self.optimizer_kwargs["args"]])


    def get_bin_limits(self, bin_index_tuple):
        """Get the bin limits corresponding to a tuple of per-dimension bin indices."""
        bounds = []
        for d in range(self.n_dims):
            index_d = bin_index_tuple[d]
            # Add a tuple (x_d_min, x_d_max) for dimension d
            bounds.append((self.bin_limits_per_dim[d][index_d], self.bin_limits_per_dim[d][index_d+1]))
        return bounds


    def get_bin_center(self, bin_index_tuple):
        """Get the bin center corresponding to a tuple of per-dimension bin indices."""
        bin_limits = self.get_bin_limits(bin_index_tuple)
        bin_center = np.array([0.5 * (x_min + x_max) for x_min,x_max in bin_limits])
        return bin_center


    def _worker_function(self, bin_index_tuple):
        """Function to optimize the target function within a set of bounds"""
        bounds = self.get_bin_limits(bin_index_tuple)
        use_optimizer_kwargs = copy(self.optimizer_kwargs)

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
        x0 = self.get_bin_center(bin_index_tuple)

        # Do the optimization and store the result
        res = None

        if self.optimizer == "minimize":
            from scipy.optimize import minimize
            try:
                res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)
            except ValueError as e:
                warnings.warn(f"{self.print_prefix} scipy.optimize.minimize returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
                use_optimizer_kwargs["method"] = "trust-constr"
                res = minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)

        elif self.optimizer == "differential_evolution":
            from scipy.optimize import differential_evolution
            res = differential_evolution(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "basinhopping":
            from scipy.optimize import basinhopping
            if not "minimizer_kwargs" in use_optimizer_kwargs:
                use_optimizer_kwargs["minimizer_kwargs"] = {}
            # use_optimizer_kwargs["minimizer_kwargs"]["bounds"] = bounds
            if "args" in use_optimizer_kwargs:
                use_optimizer_kwargs["minimizer_kwargs"]["args"] = copy(use_optimizer_kwargs["args"])
                del(use_optimizer_kwargs["args"])
            res = basinhopping(target_function_wrapper, x0, **use_optimizer_kwargs)

        elif self.optimizer == "shgo":
            from scipy.optimize import shgo
            res = shgo(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "dual_annealing":
            from scipy.optimize import dual_annealing
            res = dual_annealing(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "direct":
            from scipy.optimize import direct
            res = direct(target_function_wrapper, bounds, **use_optimizer_kwargs)

        elif self.optimizer == "iminuit":
            from iminuit import minimize as iminuit_minimize
            res = iminuit_minimize(target_function_wrapper, x0, bounds=bounds, **use_optimizer_kwargs)
            # We need to delete the iminuit.Minuit instance from the result 
            # (of type scipy.optimize.OptimizeResult), since the Minuit 
            # instance cannot be pickled and therefore would break parallelization.
            del(res["minuit"]) 

        elif self.optimizer == "diver":
            # Note: Diver should be built *without* MPI, to avoid 
            # interference with binminpy's parallelization. 
            import diver
            def diver_target(x, fcall, finish, validvector, context):
                finish = False
                if not validvector:
                    objective = 1e300
                else: 
                    objective = target_function_wrapper(x)
                return objective, fcall+1, finish
            diver_opts = copy(use_optimizer_kwargs)
            diver_opts["lowerbounds"] = [b[0] for b in bounds]
            diver_opts["upperbounds"] = [b[1] for b in bounds]
            diver_result = diver.run(diver_target, diver_opts)
            res = OptimizeResult(
                x=diver_result[1],
                fun=diver_result[0],
            )

        return res, x_points, y_points


    def _do_bin_masking(self):
        """Apply the user-provided bin masking function."""

        use_bin_indices = []
        use_bin_index_tuples = []

        if self.bin_masking is None:
            use_bin_indices = list(range(len(self.all_bin_index_tuples)))
            use_bin_index_tuples = self.all_bin_index_tuples
        else:
            bin_mask = [True]*len(self.all_bin_index_tuples)
            for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                bin_limits = self.get_bin_limits(bin_index_tuple)
                bin_centre = np.array([0.5 * (x_min + x_max) for x_min,x_max in bin_limits])
                bin_mask[i] = self.bin_masking(bin_centre, bin_limits)
            use_bin_indices = [i for i in range(self.n_bins) if bin_mask[i]]
            use_bin_index_tuples = [self.all_bin_index_tuples[i] for i in use_bin_indices]

        return use_bin_indices, use_bin_index_tuples


    def run(self):
        """Start the optimization"""

        output = {
            "x_optimal": None,
            "y_optimal": None,
            "optimal_bins": None,
            "bin_tuples": np.array(self.all_bin_index_tuples, dtype=int),
            "bin_centers": None,
            "x_optimal_per_bin": np.full((self.n_bins, self.n_dims), np.nan),
            "y_optimal_per_bin": np.full((self.n_bins,), np.inf),
            "all_optimizer_results": [None] * self.n_bins,
            "x_evals": None,
            "y_evals": None,
        }
        
        x_evals_list = []
        y_evals_list = []

        # Masking
        use_bin_indices, use_bin_index_tuples = self._do_bin_masking()
        n_tasks = len(use_bin_indices)
        print(f"{self.print_prefix} The input space is binned using {self.n_bins} bins.", flush=True)
        print(f"{self.print_prefix} After applying the bin mask we are left with {n_tasks} optimization tasks.", flush=True)

        # Carry out all the tasks in serial
        for task_index, bin_index in enumerate(use_bin_indices):
            bin_index_tuple = self.all_bin_index_tuples[bin_index]
            task_number = task_index + 1
            worker_output = self._worker_function(bin_index_tuple)

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

        if self.return_bin_centers:
            output["bin_centers"] = np.empty((self.n_bins, self.n_dims), dtype=float)
            for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                output["bin_centers"][i] = self.get_bin_center(bin_index_tuple)

        # We're done here
        return output

