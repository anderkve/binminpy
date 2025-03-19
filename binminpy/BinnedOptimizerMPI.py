import math
import numpy as np
from copy import copy
import warnings
import itertools
from mpi4py import MPI
from scipy.stats import qmc

from binminpy.BinnedOptimizer import BinnedOptimizer

class BinnedOptimizerMPI(BinnedOptimizer):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, max_processes=1, 
                 return_evals=False, return_bin_centers=True, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0, 
                 task_distribution="even", n_tasks_per_batch=1, max_tasks_per_worker=np.inf, bin_masking=None):
        """Constructor."""
        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, return_evals,
                         return_bin_centers, optima_comparison_rtol, optima_comparison_atol, bin_masking)

        task_distribution = task_distribution.lower()
        if task_distribution not in ["even", "dynamic", "mcmc"]:
            raise Exception(f"Unknown setting for argument 'task_distribution' ('{task_distribution}'). Valid options are 'even', 'dynamic' and 'mcmc'.")
        self.task_distribution = task_distribution
        self.n_tasks_per_batch = n_tasks_per_batch
        self.max_tasks_per_worker = max_tasks_per_worker


    def run(self):
        """Start the optimization using the chosen MPI task distribution scheme."""
        if self.task_distribution == "even":
            return self.run_even_task_distribution()
        elif self.task_distribution == "dynamic":
            return self.run_dynamic_task_distribution()
        elif self.task_distribution == "mcmc":
            return self.run_mcmc_task_distribution()


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
            x_optimal_per_bin = np.full((self.n_bins, self.n_dims), np.nan)
            y_optimal_per_bin = np.full((self.n_bins,), np.inf)
            x_evals_list = []
            y_evals_list = []

            for proc_dict in gathered_results:
                for task_index, result in proc_dict.items():
                    bin_index = use_bin_indices[task_index]
                    opt_result, x_points, y_points = result
                    all_optimizer_results[bin_index] = opt_result
                    x_optimal_per_bin[bin_index] = opt_result.x
                    y_optimal_per_bin[bin_index] = opt_result.fun
                    if self.return_evals:
                        x_evals_list.extend(x_points)
                        y_evals_list.extend(y_points)

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

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

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((self.n_bins, self.n_dims), dtype=float)
                for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(self.all_bin_index_tuples, dtype=int),
                "bin_centers": bin_centers,
                "x_optimal_per_bin": x_optimal_per_bin,
                "y_optimal_per_bin": y_optimal_per_bin,
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
            x_optimal_per_bin = np.full((self.n_bins, self.n_dims), np.nan)
            y_optimal_per_bin = np.full((self.n_bins,), np.inf)
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
                    opt_result, x_points, y_points = result
                    all_optimizer_results[bin_index] = opt_result
                    x_optimal_per_bin[bin_index] = opt_result.x
                    y_optimal_per_bin[bin_index] = opt_result.fun
                    if self.return_evals:
                        x_evals_list.extend(x_points)
                        y_evals_list.extend(y_points)

                # Assign new batch if available.
                if next_task_index < n_tasks:
                    batch = tasks[next_task_index: next_task_index + self.n_tasks_per_batch]
                    next_task_index += len(batch)
                    comm.send(batch, dest=sender, tag=TASK_TAG)
                else:
                    comm.send(None, dest=sender, tag=TASK_TAG)
                    num_terminated += 1

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

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

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((self.n_bins, self.n_dims), dtype=float)
                for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(self.all_bin_index_tuples, dtype=int),
                "bin_centers": bin_centers,
                "x_optimal_per_bin": x_optimal_per_bin,
                "y_optimal_per_bin": y_optimal_per_bin,
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




# ==================================================



    def run_mcmc_task_distribution(self):
        """Run the optimization using an MPI master-worker scheme, where 
        the bins to be optimized are selected by rank 0 using MCMC chains.

        Returns:
          On rank 0: a dictionary containing global optimization results.
          On other ranks: None.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        TASK_TAG = 1
        RESULT_TAG = 2
        TERMINATE_TAG = 3

        # If only one MPI process, raise an exception
        if size == 1:
            raise Exception(f"{self.print_prefix} The 'mcmc' task distribution needs more than one MPI process.")

        # Wait here until all processes are ready.
        comm.Barrier()


        # 
        # Master process
        # 
        if rank == 0:

            # Masking
            use_bin_indices, use_bin_index_tuples = self._do_bin_masking()
            max_n_bins = len(use_bin_indices)
            print(f"{self.print_prefix} The input space is binned using {self.n_bins} bins.", flush=True)
            print(f"{self.print_prefix} After applying the bin mask we are left with {max_n_bins} bins.", flush=True)

            # Boolean array for caching evaluated points.
            evaluated_mask = np.zeros(self.n_bins_per_dim, dtype=bool)

            # Set evaluated_mask to True for all bins *not* included in use_bin_index_tuples
            # so that the MCMC will never visit these bins. 
            for bin_index_tuple in use_bin_index_tuples:
                evaluated_mask[bin_index_tuple] = True
            evaluated_mask = np.invert(evaluated_mask)

            # Prepare some containers
            next_task_index = 0
            all_optimizer_results = []
            bin_tuples = []
            bin_centers = []
            x_optimal_per_bin = []
            y_optimal_per_bin = []
            x_evals_list = []
            y_evals_list = []


            #
            # Run MCMC
            #

            n_walkers = size - 1
            walkers = {}
            iterations = {}
            tasks_performed = {}
            stop_worker = {}

            # Use latin hypercube sampling to get starting points for the walkers
            lh_sampler = qmc.LatinHypercube(d=self.n_dims)
            lh_samples = lh_sampler.random(n=n_walkers)
            lh_samples = np.array(np.floor(lh_samples * np.array(self.n_bins_per_dim)), dtype=int)

            # Initialize each walker.
            for worker in range(1, size):

                stop_worker[worker] = False

                # Starting bin. Try first with the latin hypercube point,
                # and if that point is masked out, just pick a random available point.
                x0 = lh_samples[worker-1]
                if evaluated_mask[tuple(x0)] == True:
                    available_bins = np.argwhere(evaluated_mask == False)
                    x0 = available_bins[np.random.choice(available_bins.shape[0])]

                walkers[worker] = {'x': x0, 'logp': -np.inf, 'proposal': x0}
                iterations[worker] = 0
                tasks_performed[worker] = 0
                evaluated_mask[tuple(x0)] = True
                # print(f"{self.print_prefix} rank {rank}:  Sending point x = {x0} to rank {worker}", flush=True)
                comm.send(x0, dest=worker, tag=TASK_TAG)


            # Now on to the asynchronous loop, to process results as they come.
            finished = set()
            while len(finished) < n_walkers:
                status = MPI.Status()

                # Block until any worker returns a result.
                result = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker = status.Get_source()
                
                # Register this as a performed task
                tasks_performed[worker] += 1
                if tasks_performed[worker] >= self.max_tasks_per_worker:
                    stop_worker[worker] = True

                # The received result corresponds to the proposal we had sent.
                proposal = walkers[worker]['proposal']

                # Store result and extract logp_recieved
                opt_result, x_points, y_points = result

                all_optimizer_results.append(opt_result)
                bin_tuples.append(tuple(proposal))
                x_optimal_per_bin.append(opt_result.x)
                y_optimal_per_bin.append(opt_result.fun)
                if self.return_evals:
                    x_evals_list.extend(x_points)
                    y_evals_list.extend(y_points)

                logp_received = -1. * opt_result.fun


                # Perform the Metropolis test using the new log-likelihood.
                current_logp = walkers[worker]['logp']
                if np.log(np.random.rand()) < (logp_received - current_logp):
                    # Accept the new proposal.
                    walkers[worker]['x'] = proposal
                    walkers[worker]['logp'] = logp_received


                # Now attempt to propose a new move until one is found that has not been evaluated.
                n_tries = 0
                step_size = 1
                # while iterations[worker] < max_n_proposals:
                while not stop_worker[worker]:

                    n_tries += 1
                    iterations[worker] += 1

                    # Propose a move: add a random integer step from {-1, 0, 1} in each dimension.
                    new_proposal = walkers[worker]['x'] + np.random.randint(-step_size, step_size+1, self.n_dims)

                    # Occasionally jump to a random available bin
                    if n_tries % (10*self.n_dims) == 0:
                        available_bins = np.argwhere(evaluated_mask == False)
                        # If all bins are evaluated, stop *all* workers as soon as they report back
                        if len(available_bins) == 0:
                            print(f"{self.print_prefix} rank {rank}: No more available bins! Will stop all worker processes as soon as possible.")
                            stop_worker = dict.fromkeys(stop_worker, True)
                            break
                        new_proposal = available_bins[np.random.choice(available_bins.shape[0])]
                        # print(f"{self.print_prefix} rank {rank}: Random proposal jump: {walkers[worker]['x']} --> {new_proposal}")

                    # Ensure the new proposal is within the grid bounds.
                    if np.any(new_proposal < 0) or np.any(new_proposal >= self.n_bins_per_dim):
                        continue  # out-of-bounds proposals are simply rejected

                    if evaluated_mask[tuple(new_proposal)]:
                        # Already evaluated: automatically reject (walker stays at current state).
                        if n_tries % self.n_dims == 0:
                            step_size += 1
                        #     print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker}: step_size --> {step_size}")
                        # print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker}: x = {new_proposal} is an old point.", flush=True)
                        continue  # try another proposal immediately
                    else:
                        # New point: set as proposal and send for evaluation.
                        walkers[worker]['proposal'] = new_proposal
                        evaluated_mask[tuple(new_proposal)] = True
                        comm.send(new_proposal, dest=worker, tag=TASK_TAG)
                        # print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker}: Sent new point x = {new_proposal}.", flush=True)
                        break  # exit loop and wait for evaluation from the worker


                # If the walker has reached the maximum number of proposals, send termination.
                if stop_worker[worker]:
                    if worker not in finished:
                        print(f"{self.print_prefix} rank {rank}: Sending TERMINATE_TAG to rank {worker}", flush=True)
                        comm.send(None, dest=worker, tag=TERMINATE_TAG)
                        finished.add(worker)

                if iterations[worker] % 100 == 0:
                    print(f"{self.print_prefix} rank {worker}: {iterations[worker]} proposals processed.")
                if tasks_performed[worker] % 100 == 0:
                    print(f"{self.print_prefix} rank {worker}: {tasks_performed[worker]} tasks performed.")



            # 
            # After MCMC is done
            # 

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

            # Determine the global optimum.
            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for i,bin_index_tuple in enumerate(bin_tuples):
                bin_opt_result = all_optimizer_results[i]
                if bin_opt_result is not None:
                    if bin_opt_result.fun < y_opt[0]:
                        x_opt = [bin_opt_result.x]
                        y_opt = [bin_opt_result.fun]
                        optimal_bins = [bin_tuples[i]]
                    elif math.isclose(bin_opt_result.fun, y_opt[0],
                                        rel_tol=self.optima_comparison_rtol,
                                        abs_tol=self.optima_comparison_atol):
                        x_opt.append(bin_opt_result.x)
                        y_opt.append(bin_opt_result.fun)
                        optimal_bins.append(bin_tuples[i])

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((len(bin_tuples), self.n_dims), dtype=float)
                for i, bin_index_tuple in enumerate(bin_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(bin_tuples, dtype=int),
                "bin_centers": bin_centers,
                "x_optimal_per_bin": np.array(x_optimal_per_bin),
                "y_optimal_per_bin": np.array(y_optimal_per_bin),
                "all_optimizer_results": all_optimizer_results,
                "x_evals": x_evals,
                "y_evals": y_evals,
            }

            return output

        #
        # Worker process
        #
        else:
            # Worker process: receive bins to optimize until termination signal is received.
            rank = comm.Get_rank()
            status = MPI.Status()
            while True:
                data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == TERMINATE_TAG or data is None:
                    print(f"{self.print_prefix} rank {rank}: Got TERMINATE_TAG. Will stop working now", flush=True)
                    break
                bin_index_tuple = data
                result = self._worker_function(bin_index_tuple)
                print(f"{self.print_prefix} rank {rank}: Bin {bin_index_tuple} is done. Best point: x = {result[0].x}, y = {result[0].fun}", flush=True)
                comm.send(result, dest=0, tag=RESULT_TAG)
            return None



















