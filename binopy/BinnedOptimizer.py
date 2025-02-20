import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize


class BinnedOptimizer:

    def __init__(self, target_function, binning_tuples, optimizer, optimizer_arguments={}, max_processes=1):
        """Constructor"""

        self.target_function = target_function
        self.binning_tuples = binning_tuples
        self.method = method
        self.optimizer_arguments = optimizer_arguments

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]


