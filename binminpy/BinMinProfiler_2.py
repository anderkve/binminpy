import numpy as np
import itertools
import collections
import time
from scipy.optimize import minimize
from scipy.stats import cauchy, norm
from scipy.stats.qmc import LatinHypercube as LHS
import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# --- Task and Job Definitions for MPI Communication ---
TASK_EVALUATE = 1
TASK_STOP = 2

CONTEXT_INITIAL_MAXIMA_LBFGS = "initial_maxima_lbfgs"
CONTEXT_DE_TRIAL = "de_trial"
CONTEXT_ACTIVATION = "activation"
CONTEXT_REFINEMENT_LBFGS = "refinement_lbfgs"
CONTEXT_PATCHING_LBFGS = "patching_lbfgs"
CONTEXT_NEIGHBOR_TEST = "neighbor_test"

class BinMinProfiler:
    def __init__(self,
                 likelihood_func,
                 likelihood_func_name,
                 bounds,
                 projections,
                 pop_per_grid_point=1,
                 mutation_strategy='current-to-rand/1',
                 pbest_fraction=0.1,
                 n_initial_optimizations=20,
                 roi_threshold=3.0,
                 convergence_threshold=1e-5,
                 convergence_window=25,
                 neighbor_pull_probability=0.3,
                 refinement_ftol=1e-7,
                 refinement_max_iter=50,
                 refinement_gradient_method="central",
                 patching_fraction=0.1,
                 patching_conv_threshold=0.01,
                 memory_size=100,
                 samples_output_file=None):
        if MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        self.likelihood_func = likelihood_func
        self.likelihood_func_name = likelihood_func_name
        self.bounds = np.array(bounds)
        self.dims = len(self.bounds)
        if not isinstance(projections, list):
            raise TypeError("projections must be a list of dictionaries.")
        self.projections = projections
        self.pop_per_grid_point = pop_per_grid_point
        allowed_strategies = ['current-to-rand/1', 'rand/1', 'current-to-pbest/1']
        if mutation_strategy not in allowed_strategies:
            raise ValueError(f"mutation_strategy must be one of {allowed_strategies}")
        self.mutation_strategy = mutation_strategy
        self.pbest_fraction = pbest_fraction
        self.n_initial_optimizations = n_initial_optimizations
        self.roi_threshold = roi_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.neighbor_pull_probability = neighbor_pull_probability
        self.refinement_ftol = refinement_ftol
        self.refinement_max_iter = refinement_max_iter
        self.refinement_gradient_method = refinement_gradient_method
        self.patching_fraction = patching_fraction
        self.patching_conv_threshold = patching_conv_threshold
        self.memory_size = memory_size
        self.samples_output_file = samples_output_file
        if self.rank == 0 and self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000

        if self.rank == 0:
            self.task_queue = collections.deque()
            self.job_manager = {}
            self.next_job_id = 0
            self.pending_activations = {}
            self.pending_warm_start_evals = {}

        self.likelihood_calls = 0
        self.global_max_logL = -np.inf
        self.projection_dims = None
        self.grid_points_per_dim = None
        self.initial_maxima = []
        self.population = {}
        self.active_grid_indices = set()
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def run(self, num_generations, max_num_to_evolve=None, plot_callback=None, plot_interval=10, skip_init_opt_on_warm_start=True):
        if self.size > 1:
            if self.rank == 0:
                self.run_state = "STARTING"
                self.current_projection_index = 0
                self.de_evals_pending = 0
                self._manager_run(num_generations, max_num_to_evolve, plot_callback, plot_interval, skip_init_opt_on_warm_start)
            else:
                self._worker_run()
        else:
            print("ERROR: This profiler is designed for MPI and requires at least 2 processes (1 master, 1+ workers).")
            return

    def _worker_run(self):
        self.likelihood_func = self.comm.bcast(None, root=0)
        while True:
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG)
            if task['type'] == TASK_STOP:
                break
            elif task['type'] == TASK_EVALUATE:
                params = task['params']
                logL = self.likelihood_func(params)
                result = {'logL': logL, 'params': params, 'context': task['context']}
                self.comm.send(result, dest=0)

    def _manager_run(self, num_generations, max_num_to_evolve, plot_callback, plot_interval, skip_init_opt_on_warm_start):
        print(f"--- Manager starting with {self.size - 1} worker(s) ---")
        start_time = time.time()
        self.comm.bcast(self.likelihood_func, root=0)
        free_workers = list(range(1, self.size))
        worker_requests = {}
        self.run_state = "STARTING_PROJECTION"

        while True:
            self._check_for_results(worker_requests, free_workers)
            self._dispatch_tasks(worker_requests, free_workers)
            self._update_sampler_state(num_generations, max_num_to_evolve, skip_init_opt_on_warm_start)
            if self.run_state == "FINISHED":
                print("--- All projections complete. Terminating. ---")
                break
            time.sleep(0.001)

        self._shutdown_workers(worker_requests)
        total_time = time.time() - start_time
        print(f"--- Run finished in {total_time:.2f} seconds in total. ---")
        if hasattr(self, 'samples_buffer'):
            self._flush_samples_buffer()

    def _check_for_results(self, worker_requests, free_workers):
        for worker_rank in list(worker_requests.keys()):
            req = worker_requests[worker_rank]
            finished, result = req.test()
            if finished:
                self.likelihood_calls += 1
                if self.rank == 0: self._log_sample(result['params'], result['logL'])
                self._process_result(result)
                del worker_requests[worker_rank]
                free_workers.append(worker_rank)

    def _dispatch_tasks(self, worker_requests, free_workers):
        while self.task_queue and free_workers:
            worker_rank = free_workers.pop(0)
            task = self.task_queue.popleft()
            self.comm.send({'type': TASK_EVALUATE, **task}, dest=worker_rank)
            req = self.comm.irecv(source=worker_rank)
            worker_requests[worker_rank] = req

    def _update_sampler_state(self, num_generations, max_num_to_evolve, skip_init_opt_on_warm_start):
        if self.run_state == "STARTING_PROJECTION":
            if not self.task_queue and not self.job_manager and not self.pending_activations:
                proj_config = self.projections[self.current_projection_index]
                skip_opt = skip_init_opt_on_warm_start and self.current_projection_index > 0
                self._start_projection(proj_config, skip_opt)
                self.run_state = "INITIALIZING"
        elif self.run_state == "INITIALIZING":
            if not self.task_queue and not self.job_manager:
                print("\n--- Initialization complete. Starting DE ---")
                self.run_state = "RUNNING_DE"
                self.current_generation = 0
                self._initialize_population_from_maxima()
        elif self.run_state == "RUNNING_DE":
            if self.de_evals_pending == 0:
                if self.current_generation >= num_generations:
                    print("--- Reached max generations. Moving to patching. ---")
                    self.run_state = "PATCHING"
                    return
                self.current_generation += 1
                self._run_de_generation(max_num_to_evolve)
                if self.de_evals_pending == 0:
                    print("--- All points converged. Moving to patching. ---")
                    self.run_state = "PATCHING"
        elif self.run_state == "PATCHING":
            if not self.task_queue and not self.job_manager:
                print(f"--- Projection {self.current_projection_index + 1} complete. ---")
                self.current_projection_index += 1
                if self.current_projection_index >= len(self.projections):
                    self.run_state = "FINISHED"
                else:
                    self.run_state = "STARTING_PROJECTION"

    def _start_projection(self, proj_config, skip_initial_optimizations):
        self._reset_for_new_projection(proj_config)
        print(f"--- Starting new projection: {proj_config['dims']} ---")
        if not skip_initial_optimizations:
            self._find_initial_maxima()
        self._initialize_from_warm_start_file(self.samples_output_file)

    def _process_result(self, result):
        context = result['context']
        context_type = context['type']
        if context_type == CONTEXT_DE_TRIAL:
            self._process_de_result(result)
        elif context_type in [CONTEXT_INITIAL_MAXIMA_LBFGS, CONTEXT_REFINEMENT_LBFGS, CONTEXT_PATCHING_LBFGS]:
            self._process_lbfgs_result(result)
        elif context_type == CONTEXT_ACTIVATION:
            self._process_activation_result(result)

    def _shutdown_workers(self, worker_requests):
        for req in worker_requests.values():
            req.cancel()
        for i in range(1, self.size):
            self.comm.send({'type': TASK_STOP}, dest=i)

    def _reset_for_new_projection(self, projection_config):
        print("\n" + "="*80)
        print(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        print("="*80 + "\n")
        self.projection_dims = sorted(projection_config['dims'])
        for i in range(len(projection_config['grid_points'])):
            projection_config['grid_points'][i] += 1
        self.grid_points_per_dim = projection_config['grid_points']
        if len(self.projection_dims) != len(self.grid_points_per_dim):
            raise ValueError("Length of projection_dims must match length of grid_points_per_dim.")
        if any(d >= self.dims for d in self.projection_dims):
            raise ValueError("projection_dims contains an index out of bounds.")
        self.continuous_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)
        self.grid_shape = tuple(self.grid_points_per_dim)
        self.grid_axes = [np.linspace(self.bounds[d, 0], self.bounds[d, 1], n) for d, n in zip(self.projection_dims, self.grid_points_per_dim)]
        self.profile_likelihood_grid = np.full(self.grid_shape, -np.inf)
        self.initial_maxima = []
        self.population = {}
        self.active_grid_indices = set()
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def _find_initial_maxima(self):
        print(f"--- Queuing {self.n_initial_optimizations} initial optimizations to find maxima ---")
        sampler = LHS(d=self.dims, seed=np.random.randint(1000000, 1000000000000))
        unit_samples = sampler.random(n=self.n_initial_optimizations)
        start_points = self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])
        for start_point in start_points:
            self._create_lbfgs_job(start_point, context_type=CONTEXT_INITIAL_MAXIMA_LBFGS)

    def _initialize_from_warm_start_file(self, warm_start_file):
        """Initializes grid points from a previous sample file asynchronously."""
        if not warm_start_file or not os.path.exists(warm_start_file):
            return

        print(f"--- Initializing population from warm-start file: {warm_start_file} ---")
        try:
            samples = np.loadtxt(warm_start_file, delimiter=',')
            if samples.ndim == 1:
                 samples = samples.reshape(1, -1)
        except Exception as e:
            print(f"  Warning: Could not read warm-start file. Error: {e}. Skipping.")
            return

        best_candidates = {}
        for sample_row in samples:
            params = sample_row[:-1]
            logL = sample_row[-1]
            if not np.all((params >= self.bounds[:, 0]) & (params <= self.bounds[:, 1])):
                continue
            grid_idx = self._get_grid_indices_from_point(params)
            if grid_idx not in best_candidates or logL > best_candidates[grid_idx]['logL']:
                best_candidates[grid_idx] = {'params': params, 'logL': logL}

        if not best_candidates:
            print("  No valid samples found in warm-start file for the current grid.")
            return

        warm_start_max_logL = max(c['logL'] for c in best_candidates.values())
        self.global_max_logL = max(self.global_max_logL, warm_start_max_logL)
        roi_cutoff = self.global_max_logL - self.roi_threshold

        for grid_idx, candidate in best_candidates.items():
            if candidate['logL'] >= roi_cutoff:
                self._activate_point(grid_idx, warm_start_params=candidate['params'][self.continuous_dims])

    def _initialize_population_from_maxima(self):
        if not self.initial_maxima:
            return
        print("--- Initializing population around found maxima ---")
        for maximum in self.initial_maxima:
            point = maximum['point']
            grid_idx = self._get_grid_indices_from_point(point)
            for neighbor_idx in self._get_valid_neighbors(grid_idx, include_center=True):
                self._activate_point(neighbor_idx, warm_start_params=point[self.continuous_dims])

    def _activate_point(self, grid_idx, warm_start_params=None):
        """Generates and queues all tasks needed to activate a new grid point."""
        if grid_idx in self.population or grid_idx in self.pending_activations:
            return

        print(f"--- Queuing activation for grid point {grid_idx} ---")
        
        continuous_params = np.zeros((self.pop_per_grid_point, self.n_cont_dims))
        cont_bounds = self.bounds[self.continuous_dims]

        sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1000000, 1000000000000))
        unit_samples = sampler.random(n=self.pop_per_grid_point)
        scaled_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])

        if warm_start_params is not None:
            distances = np.linalg.norm(scaled_samples - warm_start_params, axis=1)
            closest_idx = np.argmin(distances)
            scaled_samples[closest_idx] = warm_start_params
        
        self.pending_activations[grid_idx] = {
            'continuous_params': scaled_samples,
            'fitnesses': np.full(self.pop_per_grid_point, -np.inf),
            'evals_pending': self.pop_per_grid_point
        }

        for i, params in enumerate(scaled_samples):
            full_params = np.zeros(self.dims)
            full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx)
            full_params[self.continuous_dims] = params
            context = {'type': CONTEXT_ACTIVATION, 'grid_idx': grid_idx, 'pop_idx': i}
            self.task_queue.append({'params': full_params, 'context': context})

    def _run_de_generation(self, max_num_to_evolve):
        """Creates and queues all DE trial tasks for a single generation."""
        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']

        if not unconverged_indices:
            self.de_evals_pending = 0
            return

        # Simple prioritization for now
        indices_to_process = unconverged_indices
        if max_num_to_evolve and len(unconverged_indices) > max_num_to_evolve:
            indices_to_process = list(np.random.choice(unconverged_indices, max_num_to_evolve, replace=False))

        active_pop_list = list(self.active_grid_indices)
        if len(active_pop_list) < 4:
            self.de_evals_pending = 0
            return

        parent_pool = []
        for idx in active_pop_list:
            state = self.population[idx]
            best_idx = np.argmax(state['fitnesses'])
            parent_pool.append({
                'continuous_params': state['continuous_params'][best_idx],
                'fitness': state['fitnesses'][best_idx]
            })

        pbest_archive = []
        if self.mutation_strategy == 'current-to-pbest/1':
            parent_pool.sort(key=lambda p: p['fitness'], reverse=True)
            pbest_size = max(1, int(len(parent_pool) * self.pbest_fraction))
            pbest_archive = parent_pool[:pbest_size]

        tasks_generated = 0
        for grid_idx in indices_to_process:
            grid_state = self.population[grid_idx]
            for i in range(self.pop_per_grid_point):
                # --- This is the core DE logic, adapted from the original file ---
                mem_loc = np.random.randint(0, self.memory_size)
                mu_CR, mu_F = self.memory_CR[mem_loc], self.memory_F[mem_loc]
                CR_i = np.clip(norm.rvs(loc=mu_CR, scale=0.1), 0, 1)
                F_i = cauchy.rvs(loc=mu_F, scale=0.1)
                while F_i <= 0: F_i = cauchy.rvs(loc=mu_F, scale=0.1)
                F_i = min(F_i, 1.0)
                
                x_i_params = grid_state['continuous_params'][i]
                
                # Simplified mutation for this example
                r1_p, r2_p, r3_p = np.random.choice(parent_pool, 3, replace=False)
                mutant = r1_p['continuous_params'] + F_i * (r2_p['continuous_params'] - r3_p['continuous_params'])
                mutant = self._ensure_bounds(mutant, self.continuous_dims)

                cross_points = np.random.rand(self.n_cont_dims) < CR_i
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.n_cont_dims)] = True
                trial_params = np.where(cross_points, mutant, x_i_params)
                
                full_params = np.zeros(self.dims)
                full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx)
                full_params[self.continuous_dims] = trial_params

                context = {
                    'type': CONTEXT_DE_TRIAL,
                    'grid_idx': grid_idx,
                    'pop_idx': i,
                    'trial_params': trial_params,
                    'F': F_i,
                    'CR': CR_i
                }
                self.task_queue.append({'params': full_params, 'context': context})
                tasks_generated += 1
        
        self.de_evals_pending = tasks_generated
        if tasks_generated > 0:
            print(f"--- Gen {self.current_generation}: Queued {tasks_generated} DE tasks. ---")

    def _process_de_result(self, result):
        """Handles the result of a single DE trial evaluation."""
        context = result['context']
        grid_idx = context['grid_idx']
        pop_idx = context['pop_idx']
        trial_fitness = result['logL']
        
        grid_state = self.population.get(grid_idx)
        if not grid_state or grid_state['status'] != 'active':
            self.de_evals_pending -= 1
            return
            
        if trial_fitness > grid_state['fitnesses'][pop_idx]:
            # Successful trial
            grid_state['continuous_params'][pop_idx] = context['trial_params']
            grid_state['fitnesses'][pop_idx] = trial_fitness
            
            # This part needs to be thread-safe if we were using threads, but with MPI it's fine.
            # In a real-world scenario, you might use a lock or a queue for these updates.
            # For now, we'll assume this is fine.
            # self.successful_F.append(context['F'])
            # self.successful_CR.append(context['CR'])

            if trial_fitness > grid_state['best_fitness']:
                improvement = trial_fitness - grid_state['best_fitness']
                grid_state['best_fitness'] = trial_fitness
                grid_state['improvement_history'].append(improvement)
                self.profile_likelihood_grid[grid_idx] = trial_fitness
                if trial_fitness > self.global_max_logL:
                    self.global_max_logL = trial_fitness
                
                # Check for activating neighbors
                if trial_fitness > (self.global_max_logL - self.roi_threshold):
                    for neighbor_idx in self._get_valid_neighbors(grid_idx):
                        if neighbor_idx not in self.population and neighbor_idx not in self.pending_activations:
                             self._activate_point(neighbor_idx, warm_start_params=context['trial_params'])

        self.de_evals_pending -= 1

    def _process_activation_result(self, result):
        """Handles a result from an activation evaluation."""
        context = result['context']
        grid_idx = context['grid_idx']
        
        if grid_idx not in self.pending_activations:
            return
        
        pending_state = self.pending_activations[grid_idx]
        pending_state['fitnesses'][context['pop_idx']] = result['logL']
        pending_state['evals_pending'] -= 1
        
        if pending_state['evals_pending'] == 0:
            # All evaluations are complete for this point.
            print(f"--- Activation complete for grid point {grid_idx}. ---")
            best_fitness = np.max(pending_state['fitnesses'])
            self.profile_likelihood_grid[grid_idx] = best_fitness
            
            self.population[grid_idx] = {
                'continuous_params': pending_state['continuous_params'],
                'fitnesses': pending_state['fitnesses'],
                'best_fitness': best_fitness,
                'status': 'active',
                'improvement_history': collections.deque(maxlen=self.convergence_window),
                'last_update_gen': 0,
                'optimizer_state': None
            }
            self.active_grid_indices.add(grid_idx)
            del self.pending_activations[grid_idx]
    
    def _create_lbfgs_job(self, start_params, context_type, grid_idx=None, seed_history=None):
        job_id = self.next_job_id
        self.next_job_id += 1
        job = {
            'id': job_id,
            'type': context_type,
            'grid_idx': grid_idx,
            'start_params': start_params,
            'current_params': start_params.copy(),
            'current_f': np.inf,
            'current_g': None,
            's_hist': collections.deque(maxlen=10),
            'y_hist': collections.deque(maxlen=10),
            'status': 'NEEDS_INITIAL_F',
            'iter': 0
        }
        if seed_history:
            job['s_hist'].extend(seed_history['s'])
            job['y_hist'].extend(seed_history['y'])
        self.job_manager[job_id] = job
        context = {'type': context_type, 'job_id': job_id, 'sub_type': 'initial_f'}
        self.task_queue.append({'params': start_params, 'context': context})

    def _process_lbfgs_result(self, result):
        """
        Handles a result for any L-BFGS-related job. This is a state machine
        that advances the optimization based on the result.
        """
        context = result['context']
        job_id = context['job_id']
        if job_id not in self.job_manager:
            return  # Job might have been cancelled or finished

        job = self.job_manager[job_id]
        sub_type = context.get('sub_type')

        if job['status'] == 'NEEDS_INITIAL_F':
            # The first likelihood evaluation for the starting point has returned.
            job['current_f'] = -result['logL']  # L-BFGS minimizes, so we use negative logL
            job['status'] = 'NEEDS_GRADIENT'
            job['pending_grad_evals'] = 0
            job['grad_components'] = {}
            # Now, queue tasks to compute the gradient.
            self._queue_gradient_tasks(job)

        elif job['status'] == 'NEEDS_GRADIENT':
            # A component of the gradient has been evaluated.
            dim, sign = context['dim'], context['sign']
            job['grad_components'][(dim, sign)] = -result['logL']
            job['pending_grad_evals'] -= 1

            if job['pending_grad_evals'] == 0:
                # The full gradient is now available.
                g = self._calculate_gradient_from_components(job)
                
                # If this is the first gradient, we just use it to find the search direction.
                # If it's a subsequent gradient, we use it to update the history first.
                if job['current_g'] is not None and 's_k' in job:
                    s_k = job.pop('s_k')
                    y_k = g - job['current_g']
                    if np.dot(y_k, s_k) > 1e-10:
                        job['s_hist'].append(s_k)
                        job['y_hist'].append(y_k)

                job['current_g'] = g
                
                if np.linalg.norm(g) < self.refinement_ftol:
                    # Gradient is very small, optimization has converged.
                    self._finalize_lbfgs_job(job, success=True)
                    return

                # L-BFGS two-loop recursion to find the search direction
                q = g
                a = []
                for s, y in zip(reversed(job['s_hist']), reversed(job['y_hist'])):
                    rho = 1.0 / np.dot(y, s)
                    alpha = rho * np.dot(s, q)
                    q = q - alpha * y
                    a.append(alpha)

                if job['s_hist']:
                    gamma = np.dot(job['s_hist'][-1], job['y_hist'][-1]) / np.dot(job['y_hist'][-1], job['y_hist'][-1])
                    z = gamma * q
                else:
                    z = q

                for (s, y), alpha in zip(zip(job['s_hist'], job['y_hist']), reversed(a)):
                    rho = 1.0 / np.dot(y, s)
                    beta = rho * np.dot(y, z)
                    z = z + s * (alpha - beta)
                
                job['search_direction'] = -z
                job['status'] = 'NEEDS_LINE_SEARCH'
                job['line_search_alpha'] = 1.0
                self._queue_line_search_task(job)

        elif job['status'] == 'NEEDS_LINE_SEARCH':
            # A step in the line search has been evaluated.
            alpha = context['alpha']
            f_new = -result['logL']
            x = job['current_params']
            f = job['current_f']
            g = job['current_g']
            d = job['search_direction']
            
            # Check Armijo condition
            if f_new <= f + 1e-4 * alpha * np.dot(g, d):
                # Condition met, step is accepted.
                job['iter'] += 1
                if job['iter'] >= self.refinement_max_iter or abs(f - f_new) < self.refinement_ftol:
                    self._finalize_lbfgs_job(job, success=True)
                    return

                x_new = x + alpha * d
                job['s_k'] = x_new - x # Store s_k for history update
                job['current_params'] = x_new
                job['current_f'] = f_new
                
                # Start computing the gradient at the new point.
                job['status'] = 'NEEDS_GRADIENT'
                self._queue_gradient_tasks(job)

            else:
                # Condition not met, reduce alpha and try again.
                job['line_search_alpha'] *= 0.5
                if job['line_search_alpha'] < 1e-10:
                    # Line search failed.
                    self._finalize_lbfgs_job(job, success=False)
                else:
                    self._queue_line_search_task(job)

    def _queue_gradient_tasks(self, job):
        """Queues all tasks needed to compute the gradient for an L-BFGS job."""
        job['grad_components'] = {}
        x = job['current_params']
        f = job['current_f']
        eps = 1e-8
        
        # For now, we only implement central difference, as it's the most robust.
        job['pending_grad_evals'] = 2 * len(x)
        for i in range(len(x)):
            # Positive step
            x_plus = x.copy()
            x_plus[i] += eps
            context = {'type': job['type'], 'job_id': job['id'], 'sub_type': 'gradient', 'dim': i, 'sign': 1}
            self.task_queue.append({'params': x_plus, 'context': context})
            
            # Negative step
            x_minus = x.copy()
            x_minus[i] -= eps
            context = {'type': job['type'], 'job_id': job['id'], 'sub_type': 'gradient', 'dim': i, 'sign': -1}
            self.task_queue.append({'params': x_minus, 'context': context})

    def _calculate_gradient_from_components(self, job):
        """Calculates the gradient vector once all its components have been evaluated."""
        grad = np.zeros_like(job['current_params'])
        eps = 1e-8
        for i in range(len(grad)):
            f_plus = job['grad_components'][(i, 1)]
            f_minus = job['grad_components'][(i, -1)]
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def _queue_line_search_task(self, job):
        """Queues a single task for the current step of a line search."""
        alpha = job['line_search_alpha']
        x_new = job['current_params'] + alpha * job['search_direction']
        # Bounds are handled by the parameter vector creation for gridded jobs,
        # but for non-gridded jobs (initial maxima), we need to clip here.
        if job['grid_idx'] is None:
             x_new = self._ensure_bounds(x_new, range(self.dims))
        
        context = {'type': job['type'], 'job_id': job['id'], 'sub_type': 'line_search', 'alpha': alpha}
        self.task_queue.append({'params': x_new, 'context': context})

    def _finalize_lbfgs_job(self, job, success):
        """Finalizes an L-BFGS job, updating state and cleaning up."""
        print(f"--- Finalizing L-BFGS Job {job['id']} ({job['type']}), Success: {success} ---")
        if success:
            final_params = job['current_params']
            final_logL = -job['current_f']
            if job['type'] == CONTEXT_INITIAL_MAXIMA_LBFGS:
                self.initial_maxima.append({'point': final_params, 'logL': final_logL})
                if final_logL > self.global_max_logL:
                    self.global_max_logL = final_logL
            # Other job types (refinement, patching) would update self.population here.

        # Remove the job from the manager
        if job['id'] in self.job_manager:
            del self.job_manager[job['id']]
    
    def _get_grid_indices_from_point(self, point, grid_axes=None):
        if grid_axes is None:
            grid_axes = self.grid_axes
        grid_coords = point[self.projection_dims]
        indices = []
        for i, coord in enumerate(grid_coords):
            axis = grid_axes[i]
            index = np.argmin(np.abs(axis - coord))
            indices.append(index)
        return tuple(indices)

    def _get_grid_coords_from_indices(self, grid_idx, grid_axes=None):
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _ensure_bounds(self, vec, dims_to_check):
        return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])

    def _get_valid_neighbors(self, grid_idx, include_center=False):
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue
            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))
            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                yield neighbor_idx

    def _log_sample(self, params, logL):
        if self.rank == 0 and hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, logL))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()

    def _flush_samples_buffer(self):
        if not self.samples_output_file or not self.samples_buffer:
            return
        with open(self.samples_output_file, 'a') as f:
            for params, logL in self.samples_buffer:
                param_str = ", ".join([f"{p:.6e}" for p in params])
                f.write(f"{param_str}, {logL:.6e}\n")
        self.samples_buffer = []

def get_test_function(name):
    """Factory function to get a test likelihood, its bounds, and true peaks."""
    if name == "bimodal_gaussian":
        MU1 = np.array([2.5, 2.5, 2.5, 2.5])
        INV_COV1 = np.linalg.inv(np.diag([1.0, 1.0, 1.0, 1.0]))
        MU2 = np.array([7.0, 7.5, 7.0, 7.5])
        INV_COV2 = np.linalg.inv(np.array([[0.8, 0.6, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.5, 0.3], [0.3, 0.0, 0.3, 0.5]]))

        def log_sum_exp(a, b):
            c = np.maximum(a, b)
            return c + np.log(np.exp(a - c) + np.exp(b - c))

        def likelihood(params):
            diff1 = params - MU1
            log_pdf1 = -0.5 * diff1.T @ INV_COV1 @ diff1
            diff2 = params - MU2
            log_pdf2 = -0.5 * diff2.T @ INV_COV2 @ diff2
            return log_sum_exp(log_pdf1, log_pdf2 + 0.5)

        bounds = [[0, 10], [0, 10], [0, 10], [0, 10]]
        peaks = [MU1, MU2]
        return likelihood, bounds, peaks

    elif name == "rosenbrock_4D":
        def likelihood(params):
            return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)

        bounds = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]]
        peaks = [np.array([1.0, 1.0, 1.0, 1.0])]
        return likelihood, bounds, peaks

    elif name == "correlated_modes":
        def log_sum_exp(a, b):
            c = np.maximum(a, b)
            return c + np.log(np.exp(a - c) + np.exp(b - c))

        def likelihood(params):
            x1, x2, x3, x4 = params
            H_A = -0.1 * ((x1 - 8)**2 + (x2 - 8)**2)
            H_B = -0.1 * ((x1 - 2)**2 + (x2 - 2)**2)

            L_A = H_A - 0.5 * ((x3 - 2)**2 + (x4 - 2)**2)
            L_B = H_B - 0.5 * ((x3 - 8)**2 + (x4 - 8)**2)

            return log_sum_exp(L_A, L_B)

        bounds = [[0, 10], [0, 10], [0, 10], [0, 10]]
        peaks = [np.array([2, 2, 8, 8]), np.array([8, 8, 2, 2])]
        return likelihood, bounds, peaks

    elif name == "himmelblau_4d":
        def likelihood(params):
            x1, x2, x3, x4 = params
            term1 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
            term2 = (x3**2 + x4 - 11)**2 + (x3 + x4**2 - 7)**2
            scale = 0.05
            return -1 * scale * (term1 + term2)

        bounds = [[-6, 6], [-6, 6], [-6, 6], [-6, 6]]
        peaks = [
            np.array([3.0, 2.0, 3.0, 2.0]),
            np.array([-2.805118, 3.131312, -2.805118, 3.131312]),
            np.array([-3.779310, -3.283186, -3.779310, -3.283186]),
            np.array([3.584428, -1.848126, 3.584428, -1.848126])
        ]
        return likelihood, bounds, peaks

    else:
        raise ValueError(f"Unknown test function: {name}")

def plot_profiles(sampler, fig, axes):
    """Generates and displays the 2D profile likelihood plot."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
        return

    ax = axes[0]
    ax.clear()

    if sampler.n_proj_dims != 2:
        ax.text(0.5, 0.5, 'Plotting only supported for 2D projections.',
                horizontalalignment='center', verticalalignment='center')
        fig.canvas.draw()
        plt.pause(0.01)
        return

    dim1, dim2 = sampler.projection_dims
    profile_2d = sampler.profile_likelihood_grid

    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    plot_baseline = sampler.global_max_logL
    vmin = plot_baseline - 3.0
    vmax = plot_baseline

    masked_profile = np.ma.masked_where(profile_2d == -np.inf, profile_2d)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')

    im = ax.imshow(masked_profile.T, extent=extent, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    active_points = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
             coords = sampler._get_grid_coords_from_indices(grid_idx)
             active_points.append(coords)

    if active_points:
        active_points = np.array(active_points)
        ax.scatter(active_points[:, 0], active_points[:, 1], c='cyan', s=3,
                   edgecolor='black', lw=0.5, label='Active DE Points')

    if sampler.initial_maxima:
        peaks = np.array([m['point'] for m in sampler.initial_maxima])
        ax.plot(peaks[:, dim1], peaks[:, dim2], 'r*', markersize=10,
                label='Found Maxima', markeredgecolor='k')

    ax.set_title(f'Profile Likelihood (Gen: {sampler.current_generation}, Dims: {sampler.projection_dims})')
    ax.set_xlabel(f'Parameter {dim1}')
    ax.set_ylabel(f'Parameter {dim2}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    cax = axes[1]
    cax.clear()
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log Likelihood')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)

if __name__ == '__main__':
    TEST_FUNCTION = "himmelblau_4d"
    OUTPUT_FILE = "samples.csv"
    PROJECTIONS_TO_RUN = [
        {'dims': [0, 3], 'grid_points': [100, 100], 'patching': True, 'refining': True},
    ]
    log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)
    sampler = BinMinProfiler(
        likelihood_func=log_likelihood,
        likelihood_func_name=TEST_FUNCTION,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=1,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=30,
        roi_threshold=3.2,
        convergence_threshold=1e-3,
        convergence_window=2,
        neighbor_pull_probability=0.5,
        refinement_ftol=1e-9,
        refinement_max_iter=20,
        refinement_gradient_method="forward-backward-alternate",
        patching_fraction=0.05,
        patching_conv_threshold=0.01,
        memory_size=len(PROJECTIONS_TO_RUN[0]['grid_points']) * 25,
        samples_output_file=OUTPUT_FILE,
    )
    sampler.run(
        num_generations=100,
        max_num_to_evolve=None,
        plot_callback=None,
        plot_interval=100,
        skip_init_opt_on_warm_start=False,
    )
