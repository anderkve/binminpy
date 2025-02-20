import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from copy import copy
import warnings


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer_kwargs={}, max_processes=1):
        """Constructor"""

        self.target_function = target_function
        self.binning_tuples = binning_tuples  # A list on the form [(x1_min, x1_max, n_bins_x1), (x2_min, x2_max, n_bins_x2), ...]
        self.optimizer_kwargs = optimizer_kwargs

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]

        if "bounds" in self.optimizer_kwargs:
            warnings.warn("BinnedOptimizer will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])


    def worker_function(self, bounds):
        """Function to optimize the target function within a set of bounds"""

        # Initial point for the optimization
        x0 = np.array([0.5 * (x_min + x_max) for x_min,x_max in bounds])

        # Do the optimization and store the result
        res = None
        try:
            res = minimize(self.target_function, x0, bounds=bounds, **self.optimizer_kwargs)
        except ValueError as e:
            warnings.warn(f"Optimization attempt returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
            modified_optimizer_kwargs = copy(self.optimizer_kwargs)
            modified_optimizer_kwargs["method"] = "trust-constr"
            res = minimize(self.target_function, x0, bounds=bounds, **modified_optimizer_kwargs)

        return res



# Bruk samme stratgi som i example_code__hyperpar_loglike_gridscan.py.
# Det burde også passe bra for MPI-parallelisering med master-worker...
