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
        self.max_processes = max_processes

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

        # Dummy example for 3 bins and a 1D function:
        all_bounds = [
            [(-2,2)], 
            [(2,4)], 
            [(4,6)],
            [(6,8)],
        ]

        # Use ProcessPoolExecutor for parallel execution
        collected_results = None
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:

            # Map the bins to the worker_function in parallel
            collected_results = executor.map(self.worker_function, all_bounds)

            # # Fill the results into the numpy array
            # for idx, result in enumerate(results):
            #     grid_results_array[idx, :] = result
            #     if idx % 100 == 0:
            #         print(f"- Done points: {idx}")

        return collected_results


# Bruk samme stratgi som i example_code__hyperpar_loglike_gridscan.py.
# Det burde også passe bra for MPI-parallelisering med master-worker...
