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
                 return_evals=False, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0,
                 task_distribution="even", n_tasks_per_batch=1, bin_masking=None):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, return_evals,
                         optima_comparison_rtol, optima_comparison_atol, bin_masking)

        task_distribution = task_distribution.lower()
        if task_distribution not in ["even", "dynamic"]:
            raise Exception(f"Unknown setting for argument 'task_distribution' ('{task_distribution}'). Valid options are 'even' and 'dynamic'.")
        self.task_distribution = task_distribution
        self.n_tasks_per_batch = n_tasks_per_batch


    def run(self):
        """Start the optimization using the chosen MPI task distribution scheme."""
        if self.task_distribution == "even":
            return self.run_even_task_distribution()
        elif self.task_distribution == "dynamic":
            return self.run_dynamic_task_distribution()


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

        # Masking
        use_bin_indices, use_bin_index_tuples = self._do_bin_masking()
        if rank == 0:
            n_tasks = len(use_bin_indices)
            print(f"{self.print_prefix} The input space is binned using {self.n_bins} bins.", flush=True)
            print(f"{self.print_prefix} After applying the bin mask we are left with {n_tasks} optimization tasks.", flush=True)

        # Distribute tasks: each task corresponds to a bin.
        my_task_indices = []
        my_bin_index_tuples = []
        for task_index, bin_index_tuple in enumerate(use_bin_index_tuples):
            if task_index % size == rank:
                my_task_indices.append(task_index)
                my_bin_index_tuples.append(bin_index_tuple)

        # Wait here until all processes are ready
        comm.Barrier()

        # Each process computes its assigned tasks.
        my_results = {}
        for task_index, bin_index_tuple in zip(my_task_indices, my_bin_index_tuples):
            task_number = task_index + 1
            result = self._worker_function(bin_index_tuple)
            my_results[task_index] = result
            print(f"{self.print_prefix} Task {task_number} with bin index tuple {bin_index_tuple} is done.", flush=True)

        # Gather all results at rank 0.
        gathered_results = comm.gather(my_results, root=0)

        if rank == 0:
            all_optimizer_results = [None] * self.n_bins
            x_evals_list = []
            y_evals_list = []
            for proc_dict in gathered_results:
                for task_index, result in proc_dict.items():
                    bin_index = use_bin_indices[task_index]
                    if self.return_evals:
                        opt_result, x_points, y_points = result
                        all_optimizer_results[bin_index] = opt_result
                        x_evals_list.append(x_points)
                        y_evals_list.append(y_points)
                    else:
                        all_optimizer_results[bin_index] = result

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
            for bin_index in range(self.n_bins):
                bin_opt_result = all_optimizer_results[bin_index]
                if bin_opt_result is not None:
                    if bin_opt_result.fun < y_opt[0]:
                        x_opt = [bin_opt_result.x]
                        y_opt = [bin_opt_result.fun]
                        optimal_bins = [self.all_bin_index_tuples[bin_index]]
                    elif math.isclose(bin_opt_result.fun, y_opt[0],
                                        rel_tol=self.optima_comparison_rtol,
                                        abs_tol=self.optima_comparison_atol):
                        x_opt.append(bin_opt_result.x)
                        y_opt.append(bin_opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[bin_index])

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


    def run_dynamic_task_distribution(self):
        """Run the optimization using an MPI master-worker scheme with task batches.

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

        # Masking
        use_bin_indices, use_bin_index_tuples = self._do_bin_masking()
        if rank == 0:
            n_tasks = len(use_bin_indices)
            print(f"{self.print_prefix} The input space is binned using {self.n_bins} bins.", flush=True)
            print(f"{self.print_prefix} After applying the bin mask we are left with {n_tasks} optimization tasks.", flush=True)

        # Wait here until all processes are ready.
        comm.Barrier()

        # If multiple MPI processes, use a master-worker pattern.
        TASK_TAG = 1
        RESULT_TAG = 2

        if rank == 0:
            # Master process.
            tasks = [(task_index, bin_index_tuple) for task_index, bin_index_tuple in enumerate(use_bin_index_tuples)]
            n_tasks = len(tasks)
            next_task_index = 0
            all_optimizer_results = [None] * self.n_bins
            x_evals_list = []
            y_evals_list = []

            # Initially send one batch to each worker.
            for worker in range(1, size):
                if next_task_index < n_tasks:
                    batch = tasks[next_task_index: next_task_index + self.n_tasks_per_batch]
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
                for task_index, result in result_dict.items():
                    bin_index = use_bin_indices[task_index]
                    if self.return_evals:
                        opt_result, x_points, y_points = result
                        all_optimizer_results[bin_index] = opt_result
                        x_evals_list.append(x_points)
                        y_evals_list.append(y_points)
                    else:
                        all_optimizer_results[bin_index] = result

                # Assign new batch if available.
                if next_task_index < n_tasks:
                    batch = tasks[next_task_index: next_task_index + self.n_tasks_per_batch]
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
            for bin_index in range(self.n_bins):
                bin_opt_result = all_optimizer_results[bin_index]
                if bin_opt_result is not None:
                    if bin_opt_result.fun < y_opt[0]:
                        x_opt = [bin_opt_result.x]
                        y_opt = [bin_opt_result.fun]
                        optimal_bins = [self.all_bin_index_tuples[bin_index]]
                    elif math.isclose(bin_opt_result.fun, y_opt[0],
                                        rel_tol=self.optima_comparison_rtol,
                                        abs_tol=self.optima_comparison_atol):
                        x_opt.append(bin_opt_result.x)
                        y_opt.append(bin_opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[bin_index])
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
                for task_index, bin_index_tuple in batch:
                    task_number = task_index + 1
                    result = self._worker_function(bin_index_tuple)
                    results[task_index] = result
                    print(f"{self.print_prefix} Task {task_number} with bin index tuple {bin_index_tuple} is done.", flush=True)
                comm.send(results, dest=0, tag=RESULT_TAG)
            return None

