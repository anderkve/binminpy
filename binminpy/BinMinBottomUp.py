import math
import numpy as np
from copy import copy
import warnings
import itertools
from mpi4py import MPI
import bisect
from itertools import product
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import OptimizeResult
from scipy.stats.qmc import LatinHypercube

from binminpy.BinMin import BinMinBase

# Module level containers (due to pickling)
_x_points_per_rank = []
_y_points_per_rank = []
_g_points_per_rank = []


class BinMinBottomUp(BinMinBase):

    def __init__(self, target_function, binning_tuples, args=(), 
                 guide_function=None, bin_check_function=None, 
                 callback=None, callback_on_rank_0=True,
                 sampler="latinhypercube", 
                 optimizer="minimize", optimizer_kwargs={},
                 sampled_parameters=None, 
                 set_eval_points=None, set_eval_points_on_rank_0=True,
                 initial_optimizer="minimize", n_initial_points=10,
                 initial_optimizer_kwargs={}, 
                 n_sampler_points_per_bin=10,
                 inherit_best_init_point_within_bin=False,
                 accept_target_below=np.inf, accept_delta_target_below=np.inf,
                 accept_guide_below=np.inf, accept_delta_guide_below=np.inf,
                 save_evals=False, return_evals=False, 
                 return_bin_results=True, return_bin_centers=True, 
                 optima_comparison_rtol=1e-9, optima_comparison_atol=0.0,
                 neighborhood_distance=1,
                 n_optim_restarts_per_bin=1, n_tasks_per_batch=1, 
                 print_progress_every_n_batch=100,
                 max_tasks_per_worker=np.inf, max_n_bins=np.inf):
        """Constructor."""

        self.print_prefix = "BinMinBottomUp:"

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_workers = size - 1

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
            if ( (accept_target_below != np.inf) or (accept_delta_target_below != np.inf)
                 or (accept_guide_below != np.inf) or (accept_delta_guide_below != np.inf) ):
                warnings.warn(f"{self.print_prefix} Since a 'bin_check_function' has been provided, the options 'accept_target_below', 'accept_delta_target_below', 'accept_guide_below' and 'accept_delta_guide_below' will be ignored.")
        self.callback = callback
        self.callback_on_rank_0 = callback_on_rank_0

        self.sampler = sampler
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.sampled_parameters = sampled_parameters
        self.set_eval_points = set_eval_points
        self.set_eval_points_on_rank_0 = set_eval_points_on_rank_0

        self.set_eval_points_from_worker = bool((self.set_eval_points is not None) and (not self.set_eval_points_on_rank_0))
        self.set_eval_points_from_rank_0 = bool((self.set_eval_points is not None) and (self.set_eval_points_on_rank_0))

        self.initial_optimizer = initial_optimizer
        self.initial_optimizer_kwargs = initial_optimizer_kwargs
        self.n_initial_points = n_initial_points
        self.n_sampler_points_per_bin = n_sampler_points_per_bin
        self.inherit_best_init_point_within_bin = inherit_best_init_point_within_bin

        self.accept_target_below = accept_target_below
        self.accept_delta_target_below = accept_delta_target_below
        if guide_function is None:
            self.accept_guide_below = self.accept_target_below
            self.accept_delta_guide_below = self.accept_delta_target_below
        else:
            self.accept_guide_below = accept_guide_below
            self.accept_delta_guide_below = accept_delta_guide_below

        self.save_evals = save_evals
        self.return_evals = return_evals
        self.return_bin_results = return_bin_results
        self.return_bin_centers = return_bin_centers

        self.optima_comparison_rtol = optima_comparison_rtol
        self.optima_comparison_atol = optima_comparison_atol

        self.neighborhood_distance = neighborhood_distance
        self.n_optim_restarts_per_bin = n_optim_restarts_per_bin
        self.n_tasks_per_batch = n_tasks_per_batch
        self.print_progress_every_n_batch = print_progress_every_n_batch
        self.max_tasks_per_worker = max_tasks_per_worker
        self.max_n_bins = max_n_bins

        # Parameters that are not listed in sampled_parameters will be optimized
        all_parameters = tuple(range(self.n_dims))
        if self.sampled_parameters is None:
            self.optimized_parameters = all_parameters
        else:
            if not isinstance(self.sampled_parameters, tuple):
                self.sampled_parameters = tuple([self.sampled_parameters])
            self.optimized_parameters = tuple(set(all_parameters).difference(self.sampled_parameters))

        self.n_sampled_dims = len(self.sampled_parameters)
        self.n_optimized_dims = len(self.optimized_parameters)

        known_samplers = ["random", "latinhypercube", "bincenter"]
        if self.sampler not in known_samplers:
            raise Exception(f"Unknown sampler '{self.sampler}'. The known samplers are {known_samplers}.")

        known_optimizers = ["minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct", "iminuit", "diver"]
        if self.optimizer not in known_optimizers:
            raise Exception(f"Unknown optimizer '{self.optimizer}'. The known optimizers are {known_optimizers}.")

        known_initial_optimizers = ["minimize", "differential_evolution"]
        if self.initial_optimizer not in known_initial_optimizers:
            raise Exception(f"Unknown initial optimizer '{self.initial_optimizer}'. The available optimizers for the initial stage are {known_initial_optimizers}.")

        if self.optimizer_kwargs == {}:
            self.optimizer_kwargs["tol"] = 1e-9
            self.optimizer_kwargs["method"] = "L-BFGS-B"

        if self.initial_optimizer_kwargs == {}:
            if self.initial_optimizer == "minimize":
                self.initial_optimizer_kwargs["tol"] = 1e-9
                self.initial_optimizer_kwargs["method"] = "L-BFGS-B"
            elif self.initial_optimizer == "differential_evolution":
                self.initial_optimizer_kwargs["popsize"] = max(15*self.n_dims, n_workers)
                self.initial_optimizer_kwargs["maxiter"] = 100
                self.initial_optimizer_kwargs["tol"] = 0.01
                self.initial_optimizer_kwargs["strategy"] = "best1bin" # "rand1bin"

        if "bounds" in self.optimizer_kwargs:
            if self.optimizer_kwargs["bounds"] is not None:
                warnings.warn(f"self.{print_prefix} The 'bounds' entry provided via the 'optimizer_kwargs' dictionary will be overridden.")
            del(self.optimizer_kwargs["bounds"])

        if "bounds" in self.initial_optimizer_kwargs:
            if self.initial_optimizer_kwargs["bounds"] is not None:
                warnings.warn(f"self.{print_prefix} The 'bounds' entry provided via the 'initial_optimizer_kwargs' dictionary will be overridden.")
            del(self.initial_optimizer_kwargs["bounds"])

        if "args" in optimizer_kwargs.keys():
            warnings.warn("The 'args' argument provided to BinMinBottomUp overrides the 'args' entry in the 'optimizer_kwargs' dictionary.")
            optimizer_kwargs["args"] = args

        # Counter used for internal bookkeeping
        self._guide_function_wrapper_calls = 0


    def _default_guide_function(self, x, y, *args):
        """Default guide function for the optimizer"""
        return y


    def _guide_function_wrapper(self, x, *args):
        global _x_points_per_rank
        global _y_points_per_rank
        global _g_points_per_rank
        self._guide_function_wrapper_calls += 1
        y = self.target_function(x, *args)
        g = self.guide_function(x, y, *args)
        # if self.return_evals or self.save_evals:
        _x_points_per_rank.append(copy(x))
        _y_points_per_rank.append(copy(y))
        _g_points_per_rank.append(copy(g))
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # print(f"rank {rank}: _guide_function_wrapper:  x = {x}  g = {g}  len(_g_points_per_rank) = {len(_g_points_per_rank)}", flush=True)
        return g


    def _sampler_func(self, n, bounds):
        """Sample a given number of points within the bounds"""
        x_lower_lims = np.array([b[0] for b in bounds])
        x_upper_lims = np.array([b[1] for b in bounds])
        if self.sampler == "random":
            sampled_points = x_lower_lims + np.random.random((n, self.n_dims)) * (x_upper_lims - x_lower_lims)
        elif self.sampler == "latinhypercube":
            lh_sampler = LatinHypercube(d=self.n_dims)
            sampled_points = x_lower_lims + lh_sampler.random(n=n) * (x_upper_lims - x_lower_lims)
        elif self.sampler == "bincenter":
            bin_center = 0.5 * (x_lower_lims + x_upper_lims)
            sampled_points = np.array([bin_center])
            # If n > 1, add random points
            if n > 1:
                random_points = x_lower_lims + np.random.random((n-1, self.n_dims)) * (x_upper_lims - x_lower_lims)
                sampled_points = np.vstack((sampled_points, random_points))
        return sampled_points        


    def _worker_function(self, bin_index_tuple, eval_points=None, return_evals=False):
        """Function to optimize the target function within a set of bounds"""

        global _x_points_per_rank
        global _y_points_per_rank
        global _g_points_per_rank

        # Run user-defined set_eval_points function on this worker process?
        if (eval_points is None) and self.set_eval_points_from_worker:
            bounds = self.get_bin_limits(bin_index_tuple)
            eval_points = self.set_eval_points(bin_index_tuple, bounds)

        x_evals_collected = []
        y_evals_collected = []

        _x_points_per_rank = []
        _y_points_per_rank = []
        _g_points_per_rank = []
        self._guide_function_wrapper_calls = 0

        # If the evaluation points have already been decided, use them
        if eval_points is not None:
            current_x_opt = None
            current_y_opt = np.inf
            current_g_opt = np.inf
            for x in eval_points:
                g = self._guide_function_wrapper(x, *self.args)
                if g < current_g_opt:
                    current_x_opt = _x_points_per_rank[-1]
                    current_y_opt = _y_points_per_rank[-1]
                    current_g_opt = g
            final_res = OptimizeResult(
                x=current_x_opt,
                fun=current_y_opt,
                guide_fun=current_g_opt,
            )
            return final_res, self.guide_function_wrapper.calls, copy(_x_points_per_rank), copy(_y_points_per_rank)

        # Since eval_points was not provided we proceed to generate points 
        # by sampling + optimization

        bounds = self.get_bin_limits(bin_index_tuple)
        use_optimizer_kwargs = copy(self.optimizer_kwargs)

        # 
        # Do the sampling
        #

        sampled_points = self._sampler_func(self.n_sampler_points_per_bin, bounds)

        # No need for optimization?
        if self.n_optimized_dims == 0:

            # Determine the best point
            current_x_opt = None
            current_y_opt = np.inf
            current_g_opt = np.inf
            for x in sampled_points:
                g = self._guide_function_wrapper(x, *self.args)
                if g < current_g_opt:
                    current_x_opt = _x_points_per_rank[-1]
                    current_y_opt = _y_points_per_rank[-1]
                    current_g_opt = g
            final_res = OptimizeResult(
                x=current_x_opt,
                fun=current_y_opt,
                guide_fun=current_g_opt,
            )
            return final_res, self._guide_function_wrapper_calls, copy(_x_points_per_rank), copy(_y_points_per_rank)

        # 
        # Do the optimization for each sampled point
        # 

        fixed_pars = {i: None for i in self.sampled_parameters}
        def wrapper_to_fix_pars(x_optimized_pars, *args):
            x = np.empty(self.n_dims)
            for i in self.sampled_parameters:
                x[i] = fixed_pars[i]
            for j, i in enumerate(self.optimized_parameters):
                x[i] = x_optimized_pars[j]
            return self._guide_function_wrapper(x, *args)

        final_res = None
        current_best_opt_pars = [self.get_bin_center(bin_index_tuple)[i] for i in self.optimized_parameters]
        for _ in range(self.n_optim_restarts_per_bin):

            for x0 in sampled_points:
                
                fixed_pars = {i: x0[i] for i in self.sampled_parameters}

                if self.inherit_best_init_point_within_bin:
                    x0_opt_init = current_best_opt_pars
                else:
                    x0_opt_init = [x0[i] for i in self.optimized_parameters]
                bounds_optimized_pars = [bounds[i] for i in self.optimized_parameters]

                if self.optimizer == "minimize":
                    try:
                        res = minimize(wrapper_to_fix_pars, x0_opt_init, bounds=bounds_optimized_pars, args=self.args, **use_optimizer_kwargs)
                    except ValueError as e:
                        warnings.warn(f"{self.print_prefix} scipy.optimize.minimize returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
                        use_optimizer_kwargs["method"] = "trust-constr"
                        res = minimize(wrapper_to_fix_pars, x0_opt_init, bounds=bounds_optimized_pars, args=self.args, **use_optimizer_kwargs)
                elif self.optimizer == "differential_evolution":
                    res = differential_evolution(wrapper_to_fix_pars, bounds_optimized_pars, args=self.args, **use_optimizer_kwargs)
                elif self.optimizer == "basinhopping":
                    from scipy.optimize import basinhopping
                    if not "minimizer_kwargs" in use_optimizer_kwargs:
                        use_optimizer_kwargs["minimizer_kwargs"] = {}
                    if "args" in use_optimizer_kwargs:
                        use_optimizer_kwargs["minimizer_kwargs"]["args"] = copy(use_optimizer_kwargs["args"])
                        del(use_optimizer_kwargs["args"])
                    res = basinhopping(wrapper_to_fix_pars, x0_opt_init, **use_optimizer_kwargs)
                elif self.optimizer == "shgo":
                    from scipy.optimize import shgo
                    res = shgo(wrapper_to_fix_pars, bounds_optimized_pars, **use_optimizer_kwargs)
                elif self.optimizer == "dual_annealing":
                    from scipy.optimize import dual_annealing
                    res = dual_annealing(wrapper_to_fix_pars, bounds_optimized_pars, **use_optimizer_kwargs)
                elif self.optimizer == "direct":
                    from scipy.optimize import direct
                    res = direct(wrapper_to_fix_pars, bounds_optimized_pars, **use_optimizer_kwargs)
                elif self.optimizer == "iminuit":
                    from iminuit import minimize as iminuit_minimize
                    res = iminuit_minimize(wrapper_to_fix_pars, x0_opt_init, bounds=bounds_optimized_pars, **use_optimizer_kwargs)
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
                            objective = wrapper_to_fix_pars(x)
                        return objective, fcall+1, finish
                    diver_opts = copy(use_optimizer_kwargs)
                    diver_opts["lowerbounds"] = [b[0] for b in bounds_optimized_pars]
                    diver_opts["upperbounds"] = [b[1] for b in bounds_optimized_pars]
                    diver_result = diver.run(diver_target, diver_opts)
                    res = OptimizeResult(
                        x=diver_result[1],
                        fun=diver_result[0],
                    )

                # The OptimizeResult.fun field should be the target function, so we create
                # a new field OptimizeResult.guide_fun for best-fit value of the guide function.
                opt_index = np.argmin(_g_points_per_rank)
                res.fun = copy(_y_points_per_rank[opt_index])
                res.guide_fun = copy(_g_points_per_rank[opt_index])

                full_x_opt = np.zeros(self.n_dims)
                full_x_opt[list(self.sampled_parameters)] = x0[list(self.sampled_parameters)]
                full_x_opt[list(self.optimized_parameters)] = res.x
                res.x = full_x_opt

                if "jac" in res:
                    jac = np.zeros(self.n_dims)
                    jac[list(self.optimized_parameters)] = res.jac
                    res.jac = jac

                if "hess_inv" in res:
                    del(res.hess_inv)

                if return_evals:
                    x_evals_collected.extend(_x_points_per_rank)
                    y_evals_collected.extend(_y_points_per_rank)

                _x_points_per_rank = []
                _y_points_per_rank = []
                _g_points_per_rank = []

                # Keep the best result from the repetitions
                if final_res is None:
                    final_res = res
                else:
                    if res.guide_fun < final_res.guide_fun:
                        final_res = res
                        current_best_opt_pars = [res.x[i] for i in self.optimized_parameters]

        return final_res, self._guide_function_wrapper_calls, x_evals_collected, y_evals_collected



    def run(self):
        """Run the optimization using an MPI master-worker scheme that first finds
        bins of local minima, and then selects new bins by "growing" outwards
        from these initial bins.

        Returns:
          On rank 0: a dictionary containing global optimization results.
          On other ranks: None.
        """

        global _x_points_per_rank
        global _y_points_per_rank
        global _g_points_per_rank

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_workers = size - 1

        # Wait here until all processes are ready.
        comm.Barrier()

        TASK_TAG = 1
        RESULT_TAG = 2
        TERMINATE_TAG = 3


        # Set some flags
        callback_from_rank_0 = bool((self.callback is not None) and (self.callback_on_rank_0))
        callback_from_worker = bool((self.callback is not None) and (not self.callback_on_rank_0))


        # 
        # Step 1: Perform initial optimization to identify all local minima
        #

        n_target_calls_total = 0
        x_evals_list = []
        y_evals_list = []

        # Limits for the full input space
        x_lower_lims = np.array([bt[0] for bt in self.binning_tuples])
        x_upper_lims = np.array([bt[1] for bt in self.binning_tuples])

        if self.initial_optimizer == "minimize":

            if rank == 0:

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
                    initial_opt_results.append((opt_result.x, opt_result.fun, opt_result.guide_fun))

                # Done with the initial optimizations, so stop all workers
                for worker_rank in range(1, n_workers+1):
                    # print(f"{self.print_prefix} rank {rank}: Telling rank {worker_rank} we are done with the initial optimization.", flush=True)
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
                        # print(f"{self.print_prefix} rank {rank}: No more initial optimization tasks for me.", flush=True)
                        break

                    # Worker process: receive optimization task, perform it, and wait at barrier
                    x0, bounds = data

                    _x_points_per_rank = []
                    _y_points_per_rank = []
                    _g_points_per_rank = []

                    use_initial_optimizer_kwargs = copy(self.use_initial_optimizer_kwargs)
                    try:
                        res = minimize(self._guide_function_wrapper, x0, bounds=bounds, args=self.args, **use_initial_optimizer_kwargs)
                    except ValueError as e:
                        warnings.warn(f"{self.print_prefix} scipy.optimize.minimize returned ValueError ({e}). Trying again with method='trust-constr'.", RuntimeWarning)
                        use_optimizer_kwargs["method"] = "trust-constr"
                        res = minimize(self._guide_function_wrapper, x0, bounds=bounds, args=self.args, **use_initial_optimizer_kwargs)

                    # The OptimizeResult.fun field should be the target function, so we create
                    # a new field OptimizeResult.guide_fun for best-fit value of the guide function.
                    opt_index = np.argmin(_g_points_per_rank)
                    res.fun = copy(_y_points_per_rank[opt_index])
                    res.guide_fun = copy(_g_points_per_rank[opt_index])

                    if (not self.return_evals) and (not self.save_evals):
                        _x_points_per_rank = []
                        _y_points_per_rank = []
                        _g_points_per_rank = []

                    return_tuple = (res, self._guide_function_wrapper_calls, copy(_x_points_per_rank), copy(_y_points_per_rank), x0)

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
                                    new_data = np.array(_x_points_per_rank)[:,i]
                                elif dset_name == "y":
                                    new_data = np.array(_y_points_per_rank)
                                dset = f[dset_name]
                                current_size = dset.shape[0]
                                new_size = new_data.shape[0]
                                dset.resize(current_size + new_size, axis=0)
                                dset[current_size: current_size + new_size] = new_data

                            # Can we get rid of the x_points and y_points?
                            if not self.return_evals:
                                return_tuple = (res, self._guide_function_wrapper_calls, [], [], x0)

                    comm.send(return_tuple, dest=0, tag=RESULT_TAG)

        elif self.initial_optimizer == "differential_evolution":

            from mpi4py.futures import MPICommExecutor
            
            # TODO: Add "if self.save_evals:" functionality for the 
            #       differential_evolution option (see above).

            # Use the full input space during the initial optimization
            bounds = [(bt[0], bt[1]) for bt in self.binning_tuples]

            _x_points_per_rank = []
            _y_points_per_rank = []
            _g_points_per_rank = []

            # with MPICommExecutor(comm, root=0) as executor:
            with MPICommExecutor(comm, root=0, use_dill=True) as executor:

                # Only rank 0 gets a non-None executor
                if rank == 0:
                    print(f"{self.print_prefix} rank {rank}: Running initial global optimization...", flush=True)

                    use_initial_optimizer_kwargs = copy(self.initial_optimizer_kwargs)
                    use_initial_optimizer_kwargs["updating"] = "deferred"
                    use_initial_optimizer_kwargs["workers"] = executor.map

                    result = differential_evolution(
                        self._guide_function_wrapper,
                        bounds,
                        args=self.args,
                        **use_initial_optimizer_kwargs,
                        # popsize=max(15*self.n_dims, n_workers),
                        # maxiter=100,
                        # tol=0.01,
                        # strategy="best1bin", # "rand1bin"
                        # updating="deferred",
                        # workers=executor.map
                    )

            print(f"{self.print_prefix} rank {rank}: Evaluated {len(_x_points_per_rank)} points during the initial optimization.", flush=True)

            # Reset counter
            self._guide_function_wrapper_calls = 0

            # Gather all points at rank 0
            x_points_gathered = comm.gather(_x_points_per_rank, root=0)
            y_points_gathered = comm.gather(_y_points_per_rank, root=0)
            g_points_gathered = comm.gather(_g_points_per_rank, root=0)

            if rank == 0:

                # Flatten the lists
                x_points = [x for sub_list in x_points_gathered for x in sub_list]
                y_points = [y for sub_list in y_points_gathered for y in sub_list]
                g_points = [g for sub_list in g_points_gathered for g in sub_list]

                # Keep track of the function calls
                n_target_calls_total += len(y_points)

                # Find the best point
                min_idx = np.argmin(g_points)
                g_min = g_points[min_idx]
                y_min = y_points[min_idx]

                # Keep all points that are acceptable
                n_pts = len(g_points)
                keep_indices = []
                for i in range(n_pts):
                    y = y_points[i]
                    g = g_points[i]

                    if self.bin_check_function is not None:
                        x = x_points[i]
                        if self.bin_check_function(OptimizeResult(x=x, fun=y, guide_fun=g), [x], [y]):
                            keep_indices.append(i)
                    elif (    (y < self.accept_target_below)
                         or (y - y_min < self.accept_delta_target_below) 
                         or (g < self.accept_guide_below)
                         or (g - g_min < self.accept_delta_guide_below) ):
                        keep_indices.append(i)

                initial_opt_results = [(copy(x_points[i]), copy(y_points[i]), copy(g_points[i])) for i in keep_indices]

            _x_points_per_rank = [] 
            _y_points_per_rank = [] 
            _g_points_per_rank = []

            del x_points_gathered
            del y_points_gathered
            del g_points_gathered

        # Wait here
        # print(f"{self.print_prefix} rank {rank}: Waiting at barrier after step 1", flush=True)
        comm.Barrier()


        # 
        # Step 2: Use optimization results to construct initial set of tasks
        #

        if rank == 0:

            # TODO: This part can be simplified! Get rid of initial_opt_tuples?

            # Collect pairs (guide function value, bin tuple for best-fit point) in a sorted list
            initial_opt_tuples = []
            current_global_ymin = np.inf
            current_global_gmin = np.inf
            for x, y, g in initial_opt_results:
                current_global_ymin = min(y, current_global_ymin)
                current_global_gmin = min(g, current_global_gmin)
                bin_index_tuple = self.get_bin_index_tuple(x)
                add_pair = (g, bin_index_tuple)
                bisect.insort(initial_opt_tuples, add_pair)

            # Start constructing the initial set of tasks
            # print(f"{self.print_prefix} rank {rank}: Growing bins from {len(initial_opt_tuples)} initial bins:\n{[bin_index_tuple for g, bin_index_tuple in initial_opt_tuples]}", flush=True)
            # print(f"{self.print_prefix} rank {rank}: Growing bins from {len(initial_opt_tuples)} points.", flush=True)
            completed_tasks = 0
            ongoing_tasks = []
            available_workers = list(range(1, n_workers+1)) 
            tasks = []
            planned_and_completed_tasks = set()

            for g_val, bin_index_tuple in initial_opt_tuples:
                if bin_index_tuple not in planned_and_completed_tasks:
                    planned_and_completed_tasks.add(bin_index_tuple)
                    tasks.append(bin_index_tuple)

            # print(f"{self.print_prefix} rank {rank}: Growing bins from {len(planned_and_completed_tasks)} initial bins:\n{planned_and_completed_tasks}", flush=True)
            print(f"{self.print_prefix} rank {rank}: Growing bins from {len(planned_and_completed_tasks)} initial bins.", flush=True)


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
                        candidate = tuple(candidate.tolist())
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
                    candidate = tuple(candidate.tolist())
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
                eval_points_per_task = [None] * use_batch_size
                if self.set_eval_points_from_rank_0:
                    for i, bin_index_tuple in enumerate(batch):
                        bounds = self.get_bin_limits(bin_index_tuple)
                        eval_points_per_task[i] = self.set_eval_points(bin_index_tuple, bounds)
                comm.send((batch, eval_points_per_task), dest=worker_rank, tag=TASK_TAG)
                tasks = tasks[len(batch):]  # Chop away the tasks that go into the batch
                ongoing_tasks.extend(batch)  # Add all the tasks in batch to the ongoing_tasks list


        #
        # Step 3: Main loop
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

            x_opt = []
            y_opt = [float('inf')]
            optimal_bins = []

            print_counter = 0
            while completed_tasks < self.max_n_bins:
                print_counter += 1

                status = MPI.Status()

                if print_counter % self.print_progress_every_n_batch == 0:
                    print(f"{self.print_prefix} rank {rank}: Completed tasks: {completed_tasks}  Currently planned tasks: {len(tasks)}  Ongoing tasks: {len(ongoing_tasks)}  Available workers: {len(available_workers)}  Target calls: {n_target_calls_total}", flush=True)
                    print_counter = 0

                # Block until any worker returns a result.
                data = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker_rank = status.Get_source()
                available_workers.append(worker_rank)

                if data is not None:

                    # 'data' is a list of (bin_index_tuple, result) pairs
                    for current_bin_index_tuple, result, user_bin_check in data:
                        opt_result, n_target_calls, x_points, y_points = result
                        # Update global optima (of the target, not the guide)?
                        if (opt_result.fun < np.min(y_opt)) and (not math.isclose(opt_result.fun, np.min(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol)):
                            x_opt = [opt_result.x]
                            y_opt = [opt_result.fun]
                            optimal_bins = [bin_index_tuple]
                        elif math.isclose(opt_result.fun, np.mean(y_opt), rel_tol=self.optima_comparison_rtol, abs_tol=self.optima_comparison_atol):
                            x_opt.append(opt_result.x)
                            y_opt.append(opt_result.fun)
                            optimal_bins.append(current_bin_index_tuple)
                        # Store some results
                        if self.return_bin_results:                        
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

                        # Task bookkeeping
                        completed_tasks += 1
                        ongoing_tasks.remove(current_bin_index_tuple)

                        # Run callback function?
                        if callback_from_rank_0:
                            self.callback(opt_result, x_points, y_points)

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
                            new_bin_tuples = collect_neighbor_bins_within_dist(current_bin_index_tuple, self.neighborhood_distance)

                            # TODO: Can implement an upper bound on the number of planned tasks here
                            if len(tasks) < np.inf:
                                for bin_index_tuple in new_bin_tuples:
                                    if len(planned_and_completed_tasks) >= self.max_n_bins:
                                        print(f"{self.print_prefix} rank {rank}: Will not plan more tasks due to the limit max_n_bins = {self.max_n_bins}", flush=True)
                                        break
                                    if bin_index_tuple not in planned_and_completed_tasks:
                                        planned_and_completed_tasks.add(bin_index_tuple)
                                        tasks.append(bin_index_tuple)

                # Now send out as many new tasks as possible
                while tasks and available_workers:
                    worker_rank = available_workers.pop(0)
                    use_batch_size = max(1, min(int(np.round(len(tasks) / n_workers)), self.n_tasks_per_batch))
                    batch = tuple(tasks[0:use_batch_size])
                    eval_points_per_task = [None] * use_batch_size
                    if self.set_eval_points_from_rank_0:
                        for i, bin_index_tuple in enumerate(batch):
                            bounds = self.get_bin_limits(bin_index_tuple)
                            eval_points_per_task[i] = self.set_eval_points(bin_index_tuple, bounds)
                    comm.send((batch, eval_points_per_task), dest=worker_rank, tag=TASK_TAG)
                    tasks = tasks[len(batch):]  # Chop away the tasks that go into the batch
                    ongoing_tasks.extend(batch)  # Add all the tasks in batch to the ongoing_tasks list

                # No more work to do? Break out of while loop
                if (not tasks) and (not ongoing_tasks):
                    break

            # Done with the given number of tasks, so stop all workers
            for worker_rank in range(1, n_workers+1):
                # print(f"{self.print_prefix} rank {rank}: Sending termination signal to rank {worker_rank}", flush=True)
                comm.send(None, dest=worker_rank, tag=TERMINATE_TAG)


            # 
            # After all the work is done
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

            # Construct outptu dict
            output = {
                "x_optimal": x_opt,
                "y_optimal": y_opt,
                "optimal_bins": np.array(optimal_bins),
                "bin_tuples": np.array(bin_tuples),
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
                batch, eval_points_per_task = data
                results = []
                x_evals_collected = []
                y_evals_collected = []
                for task_i, bin_index_tuple in enumerate(batch):

                    eval_points = eval_points_per_task[task_i]

                    # Run the worker function for this bin
                    result = self._worker_function(bin_index_tuple, eval_points=eval_points, return_evals=True)

                    # Extract results
                    opt_result, n_target_calls, x_evals, y_evals = result    
                    opt_result.guide_fun = self.guide_function(opt_result.x, opt_result.fun, *self.args)
                    
                    # Run callback function on worker process?
                    if callback_from_worker:
                        self.callback(opt_result, x_evals, y_evals)

                    # Check if this bin is interesting according to the user-defined bin_check_function
                    user_bin_check = None
                    if self.bin_check_function is not None:
                        user_bin_check = self.bin_check_function(opt_result, x_evals, y_evals)

                    # Append bin result to results list
                    if self.save_evals:
                        x_evals_collected.extend(x_evals)
                        y_evals_collected.extend(y_evals)

                    # Can we get rid of the data points now?
                    if (not self.return_evals) and (not callback_from_rank_0):
                        x_evals = []
                        y_evals = []

                    # Append bin result to results list
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
        # print(f"{self.print_prefix} rank {rank}: Waiting at the final barrier", flush=True)
        comm.Barrier()
        return output


