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

from binminpy.BinnedOptimizer import BinnedOptimizer

class BinnedOptimizerMPI(BinnedOptimizer):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, 
                         return_evals, optima_comparison_rtol, optima_comparison_atol)


    def run(self, task_distribution="even", n_tasks_per_batch=1):
        """Start the optimization using the chosen MPI task distribution scheme."""
        if task_distribution not in ["even", "dynamic"]:
            raise Exception(f"Unknown setting for argument 'task_distribution' ('{task_distribution}'). Valid options are 'even' and 'dynamic'.")
        if task_distribution == "even":
            return self.run_even_task_distribution()
        elif task_distribution == "dynamic":
            return self.run_dynamic_task_distribution(n_tasks_per_batch)


    def run_even_task_distribution(self):
        """Run the optimization using an even distribution of tasks across MPI processes.

        Returns:
          On rank 0: a dictionary containing global optimization results.
          On other ranks: None.
        """

        """Distribute the optimization tasks via MPI and collect results on rank 0."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Distribute tasks: each task corresponds to a bin.
        my_indices = []
        my_tasks = []
        for idx, bin_index in enumerate(self.all_bin_index_tuples):
            if idx % size == rank:
                my_indices.append(idx)
                my_tasks.append(bin_index)

        # Each process computes its assigned tasks.
        my_results = {}
        for idx, task in zip(my_indices, my_tasks):
            result = self._worker_function(task)
            my_results[idx] = result
            print(f"{self.print_prefix} Rank {rank}: Task {idx} is done.", flush=True)

        # Gather all results at rank 0.
        gathered_results = comm.gather(my_results, root=0)

        if rank == 0:
            all_optimizer_results = [None] * self.n_bins
            x_evals_list = []
            y_evals_list = []
            for proc_dict in gathered_results:
                for idx, result in proc_dict.items():
                    if self.return_evals:
                        opt_result, x_points, y_points = result
                        all_optimizer_results[idx] = opt_result
                        x_evals_list.append(x_points)
                        y_evals_list.append(y_points)
                    else:
                        all_optimizer_results[idx] = result

            if self.return_evals:
                x_evals = np.vstack(x_evals_list) if x_evals_list else np.zeros((0, self.n_dims))
                y_evals = np.hstack(y_evals_list) if y_evals_list else np.array([])
            else:
                x_evals = np.zeros((0, self.n_dims))
                y_evals = np.array([])

            # Identify the global optimum.
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
            return None


    def run_dynamic_task_distribution(self, n_tasks_per_batch=1):
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

        # If only one MPI process, fall back to run_even_task_distribution
        if size == 1:
            return self.run_even_task_distribution()

        # If multiple MPI processes, use a master-worker pattern.
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

