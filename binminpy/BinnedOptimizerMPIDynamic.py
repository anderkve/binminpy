import math
import numpy as np
from scipy.optimize import (
    minimize,
    differential_evolution,
    basinhopping,
    shgo,
    dual_annealing,
    direct,
    OptimizeResult
)
from copy import copy
import warnings
import itertools
from mpi4py import MPI


class BinnedOptimizerMPIDynamic:
    def __init__(self, target_function, binning_tuples, optimizer="minimize",
                 optimizer_kwargs={}, return_evals=False,
                 optima_comparison_rtol=1e-9, optima_comparison_atol=0.0):
        """Constructor.

        Parameters:
          target_function: function to optimize.
          binning_tuples: list of tuples [(min, max, n_bins), ...] for each dimension.
          optimizer: string, one of ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct"].
          optimizer_kwargs: additional keyword arguments for the optimizer.
          return_evals: if True, record evaluations.
          optima_comparison_rtol, optima_comparison_atol: tolerances for comparing optima.
        """
        self.target_function = target_function
        self.binning_tuples = binning_tuples
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.return_evals = return_evals
        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol

        self.n_dims = len(binning_tuples)
        self.n_bins_per_dim = [bt[2] for bt in binning_tuples]
        self.n_bins = np.prod(self.n_bins_per_dim)
        self.bin_limits_per_dim = [
            np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1)
            for d in range(self.n_dims)
        ]
        self.all_bin_index_tuples = list(
            itertools.product(*[range(self.n_bins_per_dim[d]) for d in range(self.n_dims)])
        )

        self.print_prefix = "BinnedOptimizerMPIDynamic:"

        known_optimizers = ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct"]
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers}.")

        if "bounds" in self.optimizer_kwargs:
            warnings.warn("BinnedOptimizerMPIDynamic will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])

        # Ensure that optimizer_kwargs["args"] is a tuple.
        if "args" in self.optimizer_kwargs:
            if not isinstance(self.optimizer_kwargs["args"], tuple):
                self.optimizer_kwargs["args"] = (self.optimizer_kwargs["args"],)

    def get_bin_limits(self, bin_index_tuple):
        """Return the bounds for a given bin index tuple."""
        bounds = []
        for d in range(self.n_dims):
            index_d = bin_index_tuple[d]
            bounds.append((self.bin_limits_per_dim[d][index_d],
                           self.bin_limits_per_dim[d][index_d+1]))
        return bounds

    def _worker_function(self, bin_index_tuple):
        """Optimize the target function within the bounds corresponding to bin_index_tuple."""
        bounds = self.get_bin_limits(bin_index_tuple)
        use_optimizer_kwargs = copy(self.optimizer_kwargs)

        # Containers for recording evaluations if requested.
        x_points = []
        y_points = []

        def target_function_wrapper(x, *args):
            y = self.target_function(x, *args)
            if self.return_evals:
                x_points.append(x)
                y_points.append(y)
            return y

        # Initial guess for optimizers that require one.
        x0 = np.array([0.5 * (x_min + x_max) for x_min, x_max in bounds])
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
            if "minimizer_kwargs" not in use_optimizer_kwargs:
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

    def run(self, n_tasks_per_batch=1):
        """Run the optimization using an MPI master-worker scheme with task batches.

        Parameters:
          n_tasks_per_batch: Number of tasks to assign per batch.
        Returns:
          On rank 0: a dictionary containing global optimization results.
          On other ranks: None.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Sequential fallback for one process.
        if size == 1:
            all_optimizer_results = [None] * self.n_bins
            x_evals_list = []
            y_evals_list = []
            for idx, task in enumerate(self.all_bin_index_tuples):
                result = self._worker_function(task)
                if self.return_evals:
                    opt_result, x_points, y_points = result
                    all_optimizer_results[idx] = opt_result
                    x_evals_list.append(x_points)
                    y_evals_list.append(y_points)
                else:
                    all_optimizer_results[idx] = result
                print(f"{self.print_prefix} Task {idx} is done.", flush=True)
            if self.return_evals:
                x_evals = np.vstack(x_evals_list) if x_evals_list else np.zeros((0, self.n_dims))
                y_evals = np.hstack(y_evals_list) if y_evals_list else np.array([])
            else:
                x_evals = np.zeros((0, self.n_dims))
                y_evals = np.array([])

            # Determine the global optimum.
            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for idx in range(self.n_bins):
                bin_opt_result = all_optimizer_results[idx]
                if bin_opt_result is not None:
                    if bin_opt_result.fun < y_opt[0]:
                        x_opt = [bin_opt_result.x]
                        y_opt = [bin_opt_result.fun]
                        optimal_bins = [self.all_bin_index_tuples[idx]]
                    elif math.isclose(bin_opt_result.fun, y_opt[0],
                                        rel_tol=self.optima_comparison_rtol,
                                        abs_tol=self.optima_comparison_atol):
                        x_opt.append(bin_opt_result.x)
                        y_opt.append(bin_opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[idx])
            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_order": self.all_bin_index_tuples,
                "all_optimizer_results": all_optimizer_results,
                "x_evals": x_evals,
                "y_evals": y_evals,
            }
            return output

        # For size > 1, use a master-worker pattern.
        TASK_TAG = 1
        RESULT_TAG = 2

        if rank == 0:
            # Master process.
            tasks = [(idx, task) for idx, task in enumerate(self.all_bin_index_tuples)]
            n_tasks = len(tasks)
            next_task_index = 0
            all_optimizer_results = [None] * self.n_bins
            x_evals_list = []
            y_evals_list = []

            # Initially send one batch to each worker.
            for worker in range(1, size):
                if next_task_index < n_tasks:
                    batch = tasks[next_task_index: next_task_index + n_tasks_per_batch]
                    next_task_index += len(batch)
                    comm.send(batch, dest=worker, tag=TASK_TAG)
                else:
                    comm.send(None, dest=worker, tag=TASK_TAG)

            num_terminated = 0
            while num_terminated < (size - 1):
                status = MPI.Status()
                result_dict = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                sender = status.Get_source()
                # Process received results.
                for task_idx, result in result_dict.items():
                    if self.return_evals:
                        opt_result, x_points, y_points = result
                        all_optimizer_results[task_idx] = opt_result
                        x_evals_list.append(x_points)
                        y_evals_list.append(y_points)
                    else:
                        all_optimizer_results[task_idx] = result

                # Assign new batch if available.
                if next_task_index < n_tasks:
                    batch = tasks[next_task_index: next_task_index + n_tasks_per_batch]
                    next_task_index += len(batch)
                    comm.send(batch, dest=sender, tag=TASK_TAG)
                else:
                    comm.send(None, dest=sender, tag=TASK_TAG)
                    num_terminated += 1

            if self.return_evals:
                x_evals = np.vstack(x_evals_list) if x_evals_list else np.zeros((0, self.n_dims))
                y_evals = np.hstack(y_evals_list) if y_evals_list else np.array([])
            else:
                x_evals = np.zeros((0, self.n_dims))
                y_evals = np.array([])

            # Determine the global optimum.
            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for idx in range(self.n_bins):
                bin_opt_result = all_optimizer_results[idx]
                if bin_opt_result is not None:
                    if bin_opt_result.fun < y_opt[0]:
                        x_opt = [bin_opt_result.x]
                        y_opt = [bin_opt_result.fun]
                        optimal_bins = [self.all_bin_index_tuples[idx]]
                    elif math.isclose(bin_opt_result.fun, y_opt[0],
                                        rel_tol=self.optima_comparison_rtol,
                                        abs_tol=self.optima_comparison_atol):
                        x_opt.append(bin_opt_result.x)
                        y_opt.append(bin_opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[idx])
            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_order": self.all_bin_index_tuples,
                "all_optimizer_results": all_optimizer_results,
                "x_evals": x_evals,
                "y_evals": y_evals,
            }
            return output

        else:
            # Worker process: receive task batches until termination signal is received.
            while True:
                batch = comm.recv(source=0, tag=TASK_TAG)
                if batch is None:
                    break
                results = {}
                for task_idx, task in batch:
                    result = self._worker_function(task)
                    results[task_idx] = result
                    print(f"{self.print_prefix} Rank {rank}: Task {task_idx} is done.", flush=True)
                comm.send(results, dest=0, tag=RESULT_TAG)
            return None
