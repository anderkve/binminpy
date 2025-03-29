import math
import numpy as np
from copy import copy
import warnings
import itertools
from mpi4py import MPI
import bisect
from itertools import product
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.stats.qmc import LatinHypercube

from binminpy.BinMin import BinMinBase

class BinMinBottomUp(BinMinBase):

    def __init__(self, target_function, binning_tuples, args=(), 
                 guide_function=None, bin_check_function=None,
                 sampler="latinhypercube", sampler_kwargs={}, 
                 optimizer="minimize", optimizer_kwargs={},
                 n_initial_points=10,
                 n_sampler_points_per_bin=10,
                 accept_target_below=-np.inf, accept_delta_target_below=0.0,
                 accept_guide_below=-np.inf, accept_delta_guide_below=0.0,
                 save_evals=False, return_evals=False, return_bin_centers=True, 
                 optima_comparison_rtol=1e-9, optima_comparison_atol=0.0,
                 n_restarts_per_bin=1, n_tasks_per_batch=1, max_tasks_per_worker=np.inf, 
                 max_n_bins=np.inf):
        """Constructor."""

        self.print_prefix = "BinMinBottomUp:"

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size == 1:
            raise Exception(f"{self.print_prefix} The 'bottomup' task distribution needs more than one MPI process.")

        super().__init__(binning_tuples)

        if not isinstance(args, tuple):
            args = tuple([args])

        self.target_function = target_function
        self.args = args

        self.guide_function = guide_function
        if self.guide_function is None:
            self.guide_function = self._default_guide_function

        self.bin_check_function = bin_check_function

        if bin_check_function is not None:
            if ( (accept_target_below != -np.inf) or (accept_delta_target_below != 0)
                 or (accept_guide_below != -np.inf) or (accept_delta_guide_below != 0) ):
                warnings.warn(f"{self.print_prefix} Since a 'bin_check_function' has been provided, the options 'accept_target_below', 'accept_delta_target_below', 'accept_guide_below' and 'accept_delta_guide_below' will be ignored.")

        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.n_initial_points = n_initial_points
        self.n_sampler_points_per_bin = n_sampler_points_per_bin

        self.accept_target_below = accept_target_below,
        self.accept_delta_target_below = accept_delta_target_below,
        self.accept_guide_below = accept_guide_below,
        self.accept_delta_guide_below = accept_delta_guide_below,

        self.save_evals = save_evals
        self.return_evals = return_evals
        self.return_bin_centers = return_bin_centers

        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol
        self.n_restarts_per_bin = n_restarts_per_bin

        self.n_tasks_per_batch = n_tasks_per_batch
        self.max_tasks_per_worker = max_tasks_per_worker
        self.max_n_bins = max_n_bins

        known_samplers = ["latinhypercube"]
        if self.sampler not in known_samplers:
            raise Exception(f"Unknown sampler '{self.sampler}'. The known samplers are {known_samplers}.")

        known_optimizers = ["minimize"]
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers}.")

        known_optimizers_and_samplers = ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct", "iminuit", "diver", "bincenter", "latinhypercube"]
        if self.sampler not in known_optimizers_and_samplers:
            raise Exception(f"Unknown sampler '{self.optimizer}'. The known optimizers are {known_optimizers_and_samplers}.")
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers_and_samplers}.")

        if "bounds" in self.sampler_kwargs:
            if self.sampler_kwargs["bounds"] is not None:
                warnings.warn("BinMin will override the 'bounds' entry provided via the 'sampler_kwargs' dictionary.")
            del(self.sampler_kwargs["bounds"])

        if "bounds" in self.optimizer_kwargs:
            if self.optimizer_kwargs["bounds"] is not None:
                warnings.warn("BinMin will override the 'bounds' entry provided via the 'optimizer_kwargs' dictionary.")
            del(self.optimizer_kwargs["bounds"])

        if "args" in sampler_kwargs.keys():
            warnings.warn("The 'args' argument provided to BinMinBottomUp overrides the 'args' entry in the 'sampler_kwargs' dictionary.")
            sampler_kwargs["args"] = args
        if "args" in optimizer_kwargs.keys():
            warnings.warn("The 'args' argument provided to BinMinBottomUp overrides the 'args' entry in the 'optimizer_kwargs' dictionary.")
            optimizer_kwargs["args"] = args


    def _default_guide_function(self, x, y, *args):
        """Default guide function for the optimizer"""
        return y


    def _worker_function(self, bin_index_tuple, return_evals=False, x0_in=None):
        """Function to optimize the target function within a set of bounds"""
        bounds = self.get_bin_limits(bin_index_tuple)
        use_optimizer_kwargs = copy(self.optimizer_kwargs)

        x_points = []
        y_points = []

        # Wrapper for the target function, to allow us to save the evaluations
        def target_function_wrapper(x, *args):
            target_function_wrapper.calls += 1
            y = self.target_function(x, *args)
            if return_evals:
                x_points.append(copy(x))
                y_points.append(copy(y))
            return y

        target_function_wrapper.calls = 0

        # Do the sampling + optimization and store the result
        final_res = None
        for run_i in range(self.n_restarts_per_bin):

            # Initial point (for optimizers that need this)
            if run_i == 0:
                if x0_in is None:
                    x0 = self.get_bin_center(bin_index_tuple)
                else:
                    x0 = x0_in
            else:
                x0 = self.get_random_point_in_bin(bin_index_tuple)

            # 
            # Do the sampling
            #

            if self.sampler == "latinhypercube":

                # Limits for the full input space
                x_lower_lims = np.array([b[0] for b in bounds])
                x_upper_lims = np.array([b[1] for b in bounds])
                
                # Do the sampling
                lh_sampler = LatinHypercube(d=self.n_dims)
                lh_x_points = x_lower_lims + lh_sampler.random(n=self.n_sampler_points_per_bin) * (x_upper_lims - x_lower_lims)

                # Determine the best point
                current_x_opt = None
                current_y_opt = np.inf
                current_g_opt = np.inf
                for x in lh_x_points:
                    y = target_function_wrapper(x, *self.args)
                    g = self.guide_function(x, y, *self.args)
                    if g < current_g_opt:
                        current_x_opt = x
                        current_y_opt = y
                        current_g_opt = g
                res = OptimizeResult(
                    x=current_x_opt,
                    fun=current_y_opt,
                    guide_fun=current_g_opt,
                )


            # 
            # TODO: Do the optimization
            # 

            # Keep the best result from the repetitions
            if final_res is None:
                final_res = res
            else:
                if res.guide_fun < final_res.guide_fun:
                    final_res = res

        return final_res, target_function_wrapper.calls, x_points, y_points



    def run(self):
        """Run the optimization using an MPI master-worker scheme that first finds
        bins of local minima, and then selects new bins by "growing" outwards
        from these initial bins.

        Returns:
          On rank 0: a dictionary containing global optimization results.
          On other ranks: None.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_workers = size - 1

        # Wait here until all processes are ready.
        comm.Barrier()

        TASK_TAG = 1
        RESULT_TAG = 2
        TERMINATE_TAG = 3


        # 
        # Step 1: Each worker finds a local minimum (from the full input space)
        #

        if rank == 0:

            n_target_calls_total = 0
            x_evals_list = []
            y_evals_list = []

            # Limits for the full input space
            x_lower_lims = np.array([bt[0] for bt in self.binning_tuples])
            x_upper_lims = np.array([bt[1] for bt in self.binning_tuples])

            # Use latin hypercube sampling to get starting points for initial optimization
            lh_sampler = LatinHypercube(d=self.n_dims)
            x0_points = list(x_lower_lims + lh_sampler.random(n=self.n_initial_points) * (x_upper_lims - x_lower_lims))

            # Use the full input space during the initial optimization
            bounds = [(bt[0], bt[1]) for bt in self.binning_tuples]

            # Send out optimization tasks
            for i, x0 in enumerate(x0_points):
                worker_rank = (i % n_workers) + 1
                x0 = x0_points[i]
                opt_task_tuple = (x0, bounds)
                comm.send(opt_task_tuple, dest=worker_rank, tag=TASK_TAG)

            # Listen for and collect results
            initial_opt_results = []
            while len(initial_opt_results) < self.n_initial_points:

                # Block until any worker returns a result.
                status = MPI.Status()
                result = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker_rank = status.Get_source()

                if result is None:
                    raise Exception(f"{self.print_prefix} Initial optimization failed for rank {worker_rank}, starting from x0 = {x0}")

                opt_result, n_target_calls, x_points, y_points, x0 = result
                print(f"{self.print_prefix} rank {rank}: Initial optimization result from rank {worker_rank}, starting from x0 = {x0}  -->  x = {opt_result.x}, y = {opt_result.fun}, guide = {opt_result.guide_fun}", flush=True)

                n_target_calls_total += n_target_calls
                if self.return_evals or self.save_evals:
                    x_evals_list.extend(x_points)
                    y_evals_list.extend(y_points)

                # Save optimization result
                initial_opt_results.append(opt_result)

            # Done with the initial optimizations, so stop all workers
            for worker_rank in range(1, n_workers+1):
                print(f"{self.print_prefix} rank {rank}: Telling rank {worker_rank} we are done with the initial optimization.", flush=True)
                comm.send(None, dest=worker_rank, tag=TERMINATE_TAG)

        else: 
            # Loop for workers listening for initial optimization tasks
            while True:
                status = MPI.Status()

                # Wait here for a new message
                data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()

                # Terminate?
                if tag == TERMINATE_TAG or data is None:
                    print(f"{self.print_prefix} rank {rank}: No more initial optimization tasks for me.", flush=True)
                    break

                # Worker process: receive optimization task, perform it, and wait at barrier
                x0, bounds = data

                x_points = []
                y_points = []
                g_points = []

                # Wrapper for the guide function, to allow us to save the evaluations
                def guide_function_wrapper(x, *args):
                    guide_function_wrapper.calls += 1
                    y = self.target_function(x, *args)
                    g = self.guide_function(x, y, *args)
                    # if self.return_evals or self.save_evals:
                    x_points.append(copy(x))
                    y_points.append(copy(y))
                    g_points.append(copy(g))
                    return g
                guide_function_wrapper.calls = 0

                try:
                    # TODO: Allow user control of the optimizer settings here
                    res = minimize(guide_function_wrapper, x0, bounds=bounds, args=self.args)
                except ValueError as e:
                    warnings.warn(f"{self.print_prefix} scipy.optimize.minimize returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
                    use_optimizer_kwargs["method"] = "trust-constr"
                    res = minimize(guide_function_wrapper, x0, bounds=bounds, args=self.args)

                # The OptimizeResult.fun field should be the target function, so we create
                # a new field OptimizeResult.guide_fun for best-fit value of the guide function.
                opt_index = np.argmin(g_points)
                res.fun = copy(y_points[opt_index])
                res.guide_fun = copy(g_points[opt_index])

                if (not self.return_evals) and (not self.save_evals):
                    x_points = []
                    y_points = []
                    g_points = []

                return_tuple = (res, guide_function_wrapper.calls, x_points, y_points, x0)

                if self.save_evals:
                    import h5py
                    hdf5_filename = f"binminpy_output_rank_{rank}.hdf5"
                    hdf5_dset_names = [f"x{i}" for i in range(self.n_dims)] + ["y"]
                    with h5py.File(hdf5_filename, 'a') as f:
                        for i,dset_name in enumerate(hdf5_dset_names):
                            if dset_name in f:
                                continue
                            else:
                                f.create_dataset(dset_name, shape=(0,), maxshape=(None,), chunks=True)

                            if dset_name[0] == "x":
                                new_data = np.array(x_points)[:,i]
                            elif dset_name == "y":
                                new_data = np.array(y_points)
                            dset = f[dset_name]
                            current_size = dset.shape[0]
                            new_size = new_data.shape[0]
                            dset.resize(current_size + new_size, axis=0)
                            dset[current_size: current_size + new_size] = new_data

                        # Can we get rid of the x_points and y_points?
                        if not self.return_evals:
                            return_tuple = (res, guide_function_wrapper.calls, [], [], x0)

                comm.send(return_tuple, dest=0, tag=RESULT_TAG)

        # Wait here
        # print(f"{self.print_prefix} rank {rank}: Waiting at barrier after step 1", flush=True)
        comm.Barrier()


        # 
        # Step 2: Use optimization results to construct initial set of tasks
        #

        if rank == 0:

            # Collect pairs (target value, bin tuple for best-fit point) in a sorted list
            initial_opt_tuples = []
            current_global_ymin = np.inf
            current_global_gmin = np.inf
            for res in initial_opt_results:
                current_global_ymin = min(res.fun, current_global_ymin)
                current_global_gmin = min(res.guide_fun, current_global_gmin)
                g_val = res.guide_fun 
                x_point = res.x
                bin_index_tuple = self.get_bin_index_tuple(x_point)
                add_pair = (g_val, bin_index_tuple)
                bisect.insort(initial_opt_tuples, add_pair)

            # Start constructing the initial set of tasks
            completed_tasks = 0
            ongoing_tasks = []
            available_workers = list(range(1, n_workers+1)) 
            tasks = []
            planned_and_completed_tasks = set()

            for g_val, bin_index_tuple in initial_opt_tuples:
                if bin_index_tuple not in planned_and_completed_tasks:
                    planned_and_completed_tasks.add(bin_index_tuple)
                    tasks.append(bin_index_tuple)

            if len(tasks) == 0:
                raise Exception(f"{self.print_prefix} No optimization tasks identified after the initial optimization. Either the initial optimization failed, or this is a bug.")

            # Now we want to add more tasks by collecting neighbors 
            # First some helper functions

            # Helper function #1 
            def generate_offsets(dim, distance):
                # Generate all possible combinations for the given dimension.
                for offset in product(range(-distance, distance + 1), repeat=dim):
                    # Skip the zero offset (which would be the input point itself).
                    if offset == (0,) * dim:
                        continue
                    if sum(abs(x) for x in offset) == distance:
                        yield offset


            # Helper function #2
            def collect_n_neighbor_bins(input_bin, num_bins):
                new_bins = []
                dim = len(input_bin)
                # Start at Manhattan distance 1 and increase until we collect enough bins.
                distance = 1
                while len(new_bins) < num_bins:
                    # Generate all offsets for the current distance.
                    for offset in generate_offsets(dim, distance):
                        candidate = np.array([input_bin[i] + offset[i] for i in range(dim)], dtype=int)
                        candidate = np.maximum(candidate, np.zeros(self.n_dims, dtype=int))
                        candidate = np.minimum(candidate, np.array(self.n_bins_per_dim, dtype=int) - 1)
                        candidate = tuple(candidate)
                        new_bins.append(candidate)
                        if len(new_bins) == num_bins:
                            break
                    distance += 1
                return new_bins


            # Helper function #3
            def collect_neighbor_bins_within_dist(input_bin, distance):                
                new_bins = []
                dim = len(input_bin)
                # Generate all offsets for the current distance.
                for offset in generate_offsets(dim, distance):
                    candidate = np.array([input_bin[i] + offset[i] for i in range(dim)], dtype=int)
                    candidate = np.maximum(candidate, np.zeros(self.n_dims, dtype=int))
                    candidate = np.minimum(candidate, np.array(self.n_bins_per_dim, dtype=int) - 1)
                    candidate = tuple(candidate)
                    new_bins.append(candidate)
                return new_bins


            # Collect some more initial tasks around the current best-fit
            if len(tasks) < n_workers:
                new_bin_tuples = collect_n_neighbor_bins(initial_opt_tuples[0][1], n_workers - len(tasks))
                for new_bin_index_tuple in new_bin_tuples:
                    if new_bin_index_tuple not in planned_and_completed_tasks:
                        planned_and_completed_tasks.add(new_bin_index_tuple)
                        tasks.append(new_bin_index_tuple)


            # Send out initial batches of tasks
            while tasks and available_workers:
                worker_rank = available_workers.pop(0)
                use_batch_size = max(1, min(int(np.round(len(tasks) / n_workers)), self.n_tasks_per_batch))
                batch = tuple(tasks[0:use_batch_size])                
                tasks = tasks[len(batch):]  # Chop away the tasks that go into the batch
                comm.send(batch, dest=worker_rank, tag=TASK_TAG)
                ongoing_tasks.extend(batch)  # Add all the tasks in batch to the ongoing_tasks list


        #
        # Step 3: Main work loop
        #

        if rank == 0:

            # Prepare some containers
            all_bin_results = []
            bin_tuples = []
            bin_centers = []
            x_optimal_per_bin = []
            y_optimal_per_bin = []
            # The other containers (x_evals_list, y_evals_list, 
            # n_target_calls_total) where created during step 1 

            print_counter = 0
            while completed_tasks < self.max_n_bins:
                print_counter += 1

                status = MPI.Status()

                if print_counter % 10 == 0:
                    print(f"{self.print_prefix} rank {rank}: Completed tasks: {completed_tasks}  Planned tasks: {len(tasks)}  Ongoing tasks: {len(ongoing_tasks)}  Available workers: {len(available_workers)}  Target calls: {n_target_calls_total}", flush=True)
                    print_counter = 0

                # Block until any worker returns a result.
                data = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker_rank = status.Get_source()
                available_workers.append(worker_rank)

                if data is not None:

                    # 'data' is a list of (bin_index_tuple, result) pairs
                    for current_bin_index_tuple, result, user_bin_check in data:
                        opt_result, n_target_calls, x_points, y_points = result
                        all_bin_results.append(opt_result)
                        bin_tuples.append(current_bin_index_tuple)
                        x_optimal_per_bin.append(opt_result.x)
                        y_optimal_per_bin.append(opt_result.fun)
                        n_target_calls_total += n_target_calls
                        if self.return_evals:
                            x_evals_list.extend(x_points)
                            y_evals_list.extend(y_points)
                        # Update current global minimum values?
                        current_global_ymin = min(opt_result.fun, current_global_ymin)
                        current_global_gmin = min(opt_result.guide_fun, current_global_gmin)

                        # If this bin is considered interesting, add neighbor bins to the task list
                        nice_neighborhood = False
                        if user_bin_check is not None:
                            nice_neighborhood = user_bin_check
                        else:
                            if (    (opt_result.fun < self.accept_target_below)
                                 or (opt_result.fun - current_global_ymin < self.accept_delta_target_below) 
                                 or (opt_result.guide_fun < self.accept_guide_below)
                                 or (opt_result.guide_fun - current_global_gmin < self.accept_delta_guide_below) ):
                                nice_neighborhood = True

                        if nice_neighborhood:
                            new_bin_tuples = collect_neighbor_bins_within_dist(current_bin_index_tuple, 1)

                            # TODO: Can implement an upper bound on the number of planned tasks here
                            if len(tasks) < np.inf:
                                for bin_index_tuple in new_bin_tuples:
                                    if bin_index_tuple not in planned_and_completed_tasks:
                                        planned_and_completed_tasks.add(bin_index_tuple)
                                        tasks.append(bin_index_tuple)

                        completed_tasks += 1
                        ongoing_tasks.remove(current_bin_index_tuple)

                # Now send out as many new tasks as possible
                while tasks and available_workers:
                    worker_rank = available_workers.pop(0)
                    use_batch_size = max(1, min(int(np.round(len(tasks) / n_workers)), self.n_tasks_per_batch))
                    batch = tuple(tasks[0:use_batch_size])                
                    tasks = tasks[len(batch):]  # Chop away the tasks that go into the batch
                    comm.send(batch, dest=worker_rank, tag=TASK_TAG)
                    ongoing_tasks.extend(batch)  # Add all the tasks in batch to the ongoing_tasks list

                # No more work to do? Break out of while loop
                if (not tasks) and (not ongoing_tasks):
                    break

            # Done with the given number of tasks, so stop all workers
            for worker_rank in range(1, n_workers+1):
                print(f"{self.print_prefix} rank {rank}: Sending termination signal to rank {worker_rank}", flush=True)
                comm.send(None, dest=worker_rank, tag=TERMINATE_TAG)


            # 
            # After all the work is done
            #

            x_evals = None
            y_evals = None
            if self.return_evals:
                x_evals = np.array(x_evals_list)
                y_evals = np.array(y_evals_list)

            # Determine the global optimum (target, not guide).
            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []
            for i,bin_index_tuple in enumerate(bin_tuples):
                bin_opt_result = all_bin_results[i]
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
                "all_bin_results": all_bin_results,
                "n_target_calls": n_target_calls_total,
                "x_evals": x_evals,
                "y_evals": y_evals,
            }


        #
        # Worker process
        #

        else:
            # Worker process: receive bins to optimize until termination signal is received.
            rank = comm.Get_rank()
            status = MPI.Status()

            # Prepare output file
            if self.save_evals:
                import h5py
                hdf5_filename = f"binminpy_output_rank_{rank}.hdf5"
                hdf5_dset_names = [f"x{i}" for i in range(self.n_dims)] + ["y"]
                with h5py.File(hdf5_filename, 'a') as f:
                    for dset_name in hdf5_dset_names:
                        if dset_name in f:
                            continue
                        else:
                            f.create_dataset(dset_name, shape=(0,), maxshape=(None,), chunks=True)

            # Main worker loop
            while True:

                # Wait here for a new message
                data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()

                # Terminate?
                if tag == TERMINATE_TAG or data is None:
                    print(f"{self.print_prefix} rank {rank}: Received termination signal", flush=True)
                    break

                # Do the tasks in the batch
                batch = data
                results = []
                x_evals_collected = []
                y_evals_collected = []
                for bin_index_tuple in batch:

                    bounds = np.array(self.get_bin_limits(bin_index_tuple))
                    x0 = self.get_bin_center(bin_index_tuple)
                    # Now run the worker function for this bin
                    result = self._worker_function(bin_index_tuple, return_evals=True, x0_in=x0)

                    # Extract results
                    opt_result, n_target_calls, x_evals, y_evals = result    
                    opt_result.guide_fun = self.guide_function(opt_result.x, opt_result.fun, *self.args)
                    
                    # Check if this bin is interesting according to the user-defined bin_check_function
                    user_bin_check = None
                    if self.bin_check_function is not None:
                        user_bin_check = self.bin_check_function(opt_result, x_evals, y_evals)

                    # Can we get rid of the data points now?
                    if (not self.return_evals) and (not self.save_evals):
                        x_evals = []
                        y_evals = []

                    # Append bin result to results list
                    x_evals_collected.extend(x_evals)
                    y_evals_collected.extend(y_evals)
                    return_result = (opt_result, n_target_calls, x_evals, y_evals)
                    results.append((bin_index_tuple, return_result, user_bin_check))

                # Write to file
                if self.save_evals:
                    with h5py.File(hdf5_filename, 'a') as f:
                        for i,dset_name in enumerate(hdf5_dset_names):
                            if dset_name[0] == "x":
                                new_data = np.array(x_evals_collected)[:,i]
                            elif dset_name == "y":
                                new_data = np.array(y_evals_collected)
                            dset = f[dset_name]
                            current_size = dset.shape[0]
                            new_size = new_data.shape[0]
                            dset.resize(current_size + new_size, axis=0)
                            dset[current_size: current_size + new_size] = new_data
    
                # Send back results for the entire batch
                # print(f"{self.print_prefix} rank {rank}: Bin {bin_index_tuple} is done. Best point: x = {result[0].x}, y = {result[0].fun}", flush=True)
                comm.send(results, dest=0, tag=RESULT_TAG)

            # This MPI process is done now
            output = None



        # All together now
        print(f"{self.print_prefix} rank {rank}: Waiting at the final barrier", flush=True)
        comm.Barrier()
        return output


