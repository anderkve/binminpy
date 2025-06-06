import math
import numpy as np
from copy import copy
import warnings
import itertools
from mpi4py import MPI
import bisect

from binminpy.BinMin import BinMin

class BinMinMPI(BinMin):

    def __init__(self, target_function, binning_tuples, optimizer="minimize", optimizer_kwargs={}, 
                 return_evals=False, return_bin_results=True, return_bin_centers=True, optima_comparison_rtol=1e-9, optima_comparison_atol=0.0,
                 n_restarts_per_bin=1, task_distribution="even", n_tasks_per_batch=1, max_tasks_per_worker=np.inf, 
                 bin_masking=None, options={}):
        """Constructor."""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        super().__init__(target_function, binning_tuples, optimizer, optimizer_kwargs, return_evals,
                         return_bin_results, return_bin_centers, optima_comparison_rtol, optima_comparison_atol, 
                         n_restarts_per_bin, bin_masking)

        task_distribution = task_distribution.lower()
        if task_distribution not in ["even", "dynamic", "mcmc"]:
            raise Exception(f"Unknown setting for argument 'task_distribution' ('{task_distribution}'). Valid options are 'even', 'dynamic' and 'mcmc'.")
        self.task_distribution = task_distribution
        self.n_tasks_per_batch = n_tasks_per_batch
        self.max_tasks_per_worker = max_tasks_per_worker
        self.options = options

        if task_distribution in ["even", "dynamic"]:
            self.init_all_bin_index_tuples()

        # Default MCMC options
        if "initial_step_size" not in self.options.keys():
            self.options["initial_step_size"] = 1
        if "n_tries_before_step_increase" not in self.options.keys():
            self.options["n_tries_before_step_increase"] = 2*self.n_dims
        if "n_tries_before_jump" not in self.options.keys():
            self.options["n_tries_before_jump"] = 100*self.n_dims
        if "always_accept_target_below" not in self.options.keys():
            self.options["always_accept_target_below"] = -np.inf
        if "always_accept_delta_target_below" not in self.options.keys():
            self.options["always_accept_delta_target_below"] = 0.0
        if "suggestion_cache_size" not in self.options.keys():
            self.options["suggestion_cache_size"] = min(max(100, 10 * self.n_dims * (size - 1)), 10000) 


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
            result = self._worker_function(bin_index_tuple, return_evals=self.return_evals)
            my_results[task_index] = result
            print(f"{self.print_prefix} Task {task_number} with bin index tuple {bin_index_tuple} is done.", flush=True)

        # Gather all results at rank 0.
        gathered_results = comm.gather(my_results, root=0)

        if rank == 0:
            all_bin_results = [None] * self.n_bins if self.return_bin_results else None
            x_optimal_per_bin = np.full((self.n_bins, self.n_dims), np.nan) if self.return_bin_results else None
            y_optimal_per_bin = np.full((self.n_bins,), np.inf) if self.return_bin_results else None
            n_target_calls_total = 0
            x_evals_list = []
            y_evals_list = []

            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for proc_dict in gathered_results:
                for task_index, result in proc_dict.items():
                    bin_index = use_bin_indices[task_index]
                    opt_result, n_target_calls, x_points, y_points = result
                    # Update global optima?
                    if (opt_result.fun < np.min(y_opt)) and (not math.isclose(opt_result.fun, np.min(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol)):
                        x_opt = [opt_result.x]
                        y_opt = [opt_result.fun]
                        optimal_bins = [bin_index_tuple]
                    elif math.isclose(opt_result.fun, np.mean(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                        x_opt.append(opt_result.x)
                        y_opt.append(opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[bin_index])
                    # Store some results
                    if self.return_bin_results:
                        all_bin_results[bin_index] = opt_result
                        x_optimal_per_bin[bin_index] = opt_result.x
                        y_optimal_per_bin[bin_index] = opt_result.fun
                    n_target_calls_total += n_target_calls
                    if self.return_evals:
                        x_evals_list.extend(x_points)
                        y_evals_list.extend(y_points)

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((self.n_bins, self.n_dims))
                for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(self.all_bin_index_tuples) if self.return_bin_results else None,
                "bin_centers": bin_centers,
                "x_optimal_per_bin": x_optimal_per_bin,
                "y_optimal_per_bin": y_optimal_per_bin,
                "all_bin_results": all_bin_results,
                "n_target_calls": n_target_calls_total,
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
            all_bin_results = [None] * self.n_bins if self.return_bin_results else None
            x_optimal_per_bin = np.full((self.n_bins, self.n_dims), np.nan) if self.return_bin_results else None
            y_optimal_per_bin = np.full((self.n_bins,), np.inf) if self.return_bin_results else None
            n_target_calls_total = 0
            x_evals_list = []
            y_evals_list = []

            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []

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
                    opt_result, n_target_calls, x_points, y_points = result
                    # Update global optima?
                    if (opt_result.fun < np.min(y_opt)) and (not math.isclose(opt_result.fun, np.min(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol)):
                        x_opt = [opt_result.x]
                        y_opt = [opt_result.fun]
                        optimal_bins = [self.all_bin_index_tuples[bin_index]]
                    elif math.isclose(opt_result.fun, np.mean(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                        x_opt.append(opt_result.x)
                        y_opt.append(opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[bin_index])
                    # Store some results
                    if self.return_bin_results:
                        all_bin_results[bin_index] = opt_result
                        x_optimal_per_bin[bin_index] = opt_result.x
                        y_optimal_per_bin[bin_index] = opt_result.fun
                    n_target_calls_total += n_target_calls
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

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((self.n_bins, self.n_dims))
                for i, bin_index_tuple in enumerate(self.all_bin_index_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(self.all_bin_index_tuples) if self.return_bin_results else None,
                "bin_centers": bin_centers,
                "x_optimal_per_bin": x_optimal_per_bin,
                "y_optimal_per_bin": y_optimal_per_bin,
                "all_bin_results": all_bin_results,
                "n_target_calls": n_target_calls_total,
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
                    result = self._worker_function(bin_index_tuple, return_evals=self.return_evals)
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
        from scipy.stats.qmc import LatinHypercube

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

            # System for suggestion cache
            bin_suggestion_cache = []
            suggestion_cache_size = self.options["suggestion_cache_size"]
            def add_suggestion(y_val, bin_index_tuple):
                # Clear out-of-date suggestions:
                for i in range(len(bin_suggestion_cache))[::-1]:
                    if evaluated_mask[tuple(bin_suggestion_cache[i][1])]:
                        bin_suggestion_cache.pop(i)
                new_pair = (y_val, bin_index_tuple)
                # print(f"DEBUG: Adding new suggestion: {new_pair}", flush=True)
                bisect.insort(bin_suggestion_cache, new_pair)
                if len(bin_suggestion_cache) > suggestion_cache_size:
                    bin_suggestion_cache.pop()
                add_suggestion.max_y = bin_suggestion_cache[-1][0]
            add_suggestion.max_y = np.inf

            def get_suggestion():
                while True:
                    if len(bin_suggestion_cache) == 0:
                        return None
                    y_val, bin_index_tuple = bin_suggestion_cache.pop(0)
                    if evaluated_mask[tuple(bin_index_tuple)]:
                        continue
                    else:
                        return np.array(bin_index_tuple)

            recent_y_cache = []
            def add_to_recent_y_cache(y):
                recent_y_cache.insert(0,y)
                if len(recent_y_cache) > suggestion_cache_size:
                    recent_y_cache.pop()


            # Prepare some containers
            next_task_index = 0
            all_bin_results = []
            bin_tuples = []
            bin_centers = []
            x_optimal_per_bin = []
            y_optimal_per_bin = []
            n_target_calls_total = 0
            x_evals_list = []
            y_evals_list = []

            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []

            current_global_ymin = np.inf

            #
            # Run MCMC
            #

            n_workers = size - 1
            n_walkers = n_workers
            n_available_bins = max_n_bins

            # Lists to keep track of each walker
            walkers = [{"bin": None, "logp": None, "x": None, "logp_history": []}] * n_walkers

            # Dicts to keep track of each MPI worker
            worker_ranks = list(range(1, n_workers+1))
            tasks_performed = dict.fromkeys(worker_ranks, 0)
            stop_worker = dict.fromkeys(worker_ranks, False)

            # Use latin hypercube sampling to get starting points for the walkers
            lh_sampler = LatinHypercube(d=self.n_dims)
            lh_samples = lh_sampler.random(n=n_walkers)
            lh_samples = np.array(np.floor(lh_samples * np.array(self.n_bins_per_dim)), dtype=int)

            # Initialize each walker.
            for walker_index in range(0, n_walkers):
                # Rank 0 is the master process, so worker rank = walker index + 1
                worker_rank = walker_index + 1

                # Starting bin. Try first with the latin hypercube point,
                # and if that point is masked out, just pick a random available point.
                bin_indices = lh_samples[walker_index]
                if evaluated_mask[tuple(bin_indices)] == True:
                    available_bins = np.argwhere(evaluated_mask == False)
                    bin_indices = available_bins[np.random.choice(available_bins.shape[0])]

                walkers[walker_index] = {"bin": bin_indices, "logp": -np.inf, "x": self.get_bin_center(bin_indices), "logp_history": []}
                evaluated_mask[tuple(bin_indices)] = True
                n_available_bins -= 1
                print(f"{self.print_prefix} rank {rank}: Sending bin {bin_indices} to rank {worker_rank}", flush=True)
                comm.send(([bin_indices], None), dest=worker_rank, tag=TASK_TAG)

            # Now on to the asynchronous loop, to process results as they come.
            finished = set()
            while len(finished) < n_walkers:
                status = MPI.Status()

                print(f"{self.print_prefix} rank {rank}: Current global ymin: {current_global_ymin}  add_suggestion.max_y: {add_suggestion.max_y}  len(bin_suggestion_cache): {len(bin_suggestion_cache)}", flush=True)

                # Block until any worker returns a result.
                result_tuples = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker_rank = status.Get_source()
                walker_index = worker_rank - 1
                n_results = len(result_tuples)

                # Register the performed tasks
                tasks_performed[worker_rank] += n_results
                if tasks_performed[worker_rank] >= self.max_tasks_per_worker:
                    stop_worker[worker_rank] = True

                # Store all results received and extract the logp values
                logp_vals = []
                x_vals = []
                for proposal, result in result_tuples:
                    opt_result, n_target_calls, x_points, y_points = result
                    # Update global optima?
                    if (opt_result.fun < np.min(y_opt)) and (not math.isclose(opt_result.fun, np.min(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol)):
                        x_opt = [opt_result.x]
                        y_opt = [opt_result.fun]
                        optimal_bins = [bin_index_tuple]
                    elif math.isclose(opt_result.fun, np.mean(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                        x_opt.append(opt_result.x)
                        y_opt.append(opt_result.fun)
                        optimal_bins.append(self.all_bin_index_tuples[bin_index])
                    # Store some results
                    if self.return_bin_results:
                        all_bin_results.append(opt_result)
                        bin_tuples.append(tuple(proposal))
                        x_optimal_per_bin.append(opt_result.x)
                        y_optimal_per_bin.append(opt_result.fun)
                    n_target_calls_total += n_target_calls
                    if self.return_evals:
                        x_evals_list.extend(x_points)
                        y_evals_list.extend(y_points)
                    logp_vals.append(-1. * opt_result.fun)
                    x_vals.append(opt_result.x)
                    # Update current global best-fit value?
                    if opt_result.fun < current_global_ymin:
                        current_global_ymin = opt_result.fun
                    # Add to recent_y_cache
                    add_to_recent_y_cache(opt_result.fun)
                logp_vals = np.array(logp_vals)
                x_vals = np.array(x_vals)

                # Now order the results according to their logp value (highest first)
                # and start testing whether or not to accept the proposed moves
                logp_ordering = np.argsort(logp_vals)[::-1]
                logp_vals = logp_vals[logp_ordering]
                x_vals = x_vals[logp_ordering]
                proposals = [result_tuples[i][0] for i in logp_ordering]

                for proposal, proposal_logp, x_val in zip(proposals, logp_vals, x_vals):
                    current_logp = walkers[walker_index]['logp']

                    # Will the new proposal be accepted due to the user-defined threshold?
                    accepted_by_user_threshold = False
                    if ( (proposal_logp > -self.options["always_accept_target_below"]) 
                          or (-current_global_ymin - proposal_logp < self.options["always_accept_delta_target_below"]) ):
                        accepted_by_user_threshold = True

                    # Add available neighbor bins to cached suggestions?
                    y_val = -proposal_logp
                    if (accepted_by_user_threshold) or (y_val < np.median(recent_y_cache)):
                        for dim in range(self.n_dims):
                            for shift in [-1,1]:
                                new_index = proposal[dim] + shift
                                if (new_index < 0) or (new_index >= self.n_bins_per_dim[dim]):
                                    continue
                                neighbor_bin_tuple = np.array(proposal)
                                neighbor_bin_tuple[dim] = new_index
                                if not evaluated_mask[tuple(neighbor_bin_tuple)]:
                                    add_suggestion(y_val, tuple(neighbor_bin_tuple))

                    # Accept or reject proposal
                    if (accepted_by_user_threshold) or (np.log(np.random.rand()) < (proposal_logp - current_logp)):
                        walkers[walker_index]["bin"] = proposal
                        walkers[walker_index]["logp"] = proposal_logp
                        walkers[walker_index]["x"] = x_val
                        # walkers[walker_index]["logp_history"].insert(0, proposal_logp)
                        # Since this move was accepted, skip the rest
                        break


                # # Has this walker stagnated?
                # logp_history_size = 100
                # walkers[walker_index]["logp_history"].insert(0, proposal_logp)
                # if len(walkers[walker_index]["logp_history"]) > logp_history_size:
                #     walkers[walker_index]["logp_history"].pop()
                # logp_hist = np.array(walkers[walker_index]["logp_history"])
                # logp_increase = np.array([logp_hist[i] - logp_hist[i+1] for i in range(len(logp_hist)-1)])
                # avg_logp_increase = np.mean(logp_increase)
                # if (avg_logp_increase < -1.0) and (len(logp_hist) == logp_history_size):
                #     print(f"DEBUG: Resetting walker {walker_index}.  avg_logp_increase: {avg_logp_increase}")
                #     available_bins = np.argwhere(evaluated_mask == False)
                #     bin_indices = available_bins[np.random.choice(available_bins.shape[0])]
                #     walkers[walker_index] = {"bin": bin_indices, "logp": -np.inf, "x": self.get_bin_center(bin_indices), "logp_history": []}


                # Walker move done, now collect a batch of new proposal steps
                proposal_batch = []
                n_tries = 0
                step_size = self.options["initial_step_size"]
                # Larger step size when far away?
                if len(walkers[walker_index]["logp_history"]) > 0: 
                    walker_ymin = -1 * np.max(walkers[walker_index]["logp_history"])
                    if walker_ymin > (current_global_ymin + self.options["always_accept_delta_target_below"]):
                        step_size += 2
                # Start collection
                while (not stop_worker[worker_rank]):

                    if n_available_bins <= 0:
                        print(f"{self.print_prefix} rank {rank}: No more available bins! Will stop all worker processes as soon as possible.", flush=True)
                        stop_worker = dict.fromkeys(stop_worker, True)
                        break

                    while len(proposal_batch) < self.n_tasks_per_batch:

                        n_tries += 1

                        # Propose a move:
                        # Initially 33% chance for 0
                        new_proposal = walkers[walker_index]["bin"] + np.random.randint(-step_size, step_size+1, self.n_dims)
                        new_proposal = np.maximum(new_proposal, np.zeros(self.n_dims, dtype=int))
                        new_proposal = np.minimum(new_proposal, np.array(self.n_bins_per_dim, dtype=int) - 1)

                        # Initially 50% chance for 0
                        # new_proposal = walkers[walker_index]["bin"] + np.random.choice([-1,1], self.n_dims) * np.random.randint(0, step_size+1, self.n_dims)
                        # new_proposal = np.maximum(new_proposal, np.zeros(self.n_dims, dtype=int))
                        # new_proposal = np.minimum(new_proposal, np.array(self.n_bins_per_dim, dtype=int) - 1)

                        # Should the step size be increased?
                        if n_tries % self.options["n_tries_before_step_increase"] == 0:
                            step_size += 1
                            print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker_rank}: step_size --> {step_size}  n_tries: {n_tries}" , flush=True)

                        # Should we jump to a suggested bin?
                        if n_tries % self.options["n_tries_before_jump"] == 0:
                            suggestion = get_suggestion()
                            if suggestion is not None:
                                new_proposal = suggestion
                                print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker_rank}: Doing huge jump to bin {new_proposal}  n_tries: {n_tries}", flush=True)

                        # Check the proposed point is not already evaluated
                        if evaluated_mask[tuple(new_proposal)]:
                            continue

                        # OK, proposal can be added to the batch
                        # print(f"{self.print_prefix} rank {rank}: Finding proposal for rank {worker_rank}: OK! Proposal {new_proposal} added to task bach! n_tries: {n_tries}.", flush=True)
                        proposal_batch.append(new_proposal)
                        n_tries = 0

                    # Done collecting proposals

                    # Have we decided to stop this worker?
                    if stop_worker[worker_rank]:
                        break

                    # Now we have a batch that we can send to the worker
                    # First, set the evaluated_mask to True for all the proposals
                    for proposal in proposal_batch:
                        evaluated_mask[tuple(new_proposal)] = True
                        n_available_bins -= 1
                    # Pass along an initial point?
                    x0_in = None
                    if self.options["inherit_min_coords"]:
                        x0_in = walkers[walker_index]["x"]
                    # Send the batch
                    comm.send((proposal_batch, x0_in), dest=worker_rank, tag=TASK_TAG)
                    # Now break the outer while loop and wait for message from the worker
                    break

                # If we have decided to stop the worker, send a message with TERMINATE_TAG
                if stop_worker[worker_rank]:
                    if worker_rank not in finished:
                        print(f"{self.print_prefix} rank {rank}: Sending TERMINATE_TAG to rank {worker_rank}", flush=True)
                        comm.send(None, dest=worker_rank, tag=TERMINATE_TAG)
                        finished.add(worker_rank)

                if tasks_performed[worker_rank] % 100 == 0:
                    print(f"{self.print_prefix} rank {worker_rank}: {tasks_performed[worker_rank]} tasks performed.", flush=True)


            # 
            # After MCMC is done
            # 

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

            bin_centers = None
            if self.return_bin_centers:
                bin_centers = np.empty((len(bin_tuples), self.n_dims))
                for i, bin_index_tuple in enumerate(bin_tuples):
                    bin_centers[i] = self.get_bin_center(bin_index_tuple)

            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": optimal_bins,
                "bin_tuples": np.array(bin_tuples) if self.return_bin_results else None,
                "bin_centers": bin_centers,
                "x_optimal_per_bin": np.array(x_optimal_per_bin),
                "y_optimal_per_bin": np.array(y_optimal_per_bin),
                "all_bin_results": all_bin_results,
                "n_target_calls": n_target_calls_total,
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

                # Terminate?
                if tag == TERMINATE_TAG or data is None:
                    print(f"{self.print_prefix} rank {rank}: Got TERMINATE_TAG. Will stop working now", flush=True)
                    break

                # Do the requested tasks
                task_batch, x0_in = data
                results = []
                for bin_index_tuple in task_batch:
                    # Extract usabel coordinates from x0_in.
                    # Remaining coordinates are set using the bin center
                    if x0_in is not None:
                        bounds = np.array(self.get_bin_limits(bin_index_tuple))
                        bin_center = self.get_bin_center(bin_index_tuple)
                        mask = (x0_in >= bounds[:, 0]) & (x0_in <= bounds[:, 1])
                        for i,mask_val in enumerate(mask):
                            if mask_val == False:
                                x0_in[i] = bin_center[i]
                        # print(f"DEBUG: x0_in = {x0_in}", flush=True)
                    # Now run the worker function for this bin
                    result = self._worker_function(bin_index_tuple, return_evals=self.return_evals, x0_in=x0_in)
                    print(f"{self.print_prefix} rank {rank}: Bin {bin_index_tuple} is done. Best point: x = {result[0].x}, y = {result[0].fun}", flush=True)
                    results.append((bin_index_tuple, result))
                comm.send(results, dest=0, tag=RESULT_TAG)
            return None

