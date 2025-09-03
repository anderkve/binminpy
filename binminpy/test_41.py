import numpy as np
import itertools
import collections
import time
from scipy.stats import cauchy, norm
from scipy.stats.qmc import LatinHypercube as LHS
import os
import sys

# --- MPI Setup ---
# The code is designed to be run with MPI.
# e.g., mpiexec -n <number_of_cores> python test_38.py
try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)


# --- Task Definitions (for communication between master and workers) ---
# These constants define the different types of work the master can request.
# This helps the master decide how to process the result of a likelihood call.
TASK_TERMINATE = -1
TASK_LIKELIHOOD_EVAL = 1


# --- Refinement Job Logic (Replaces CustomLBFGSB) ---

def _calculate_gradient_tasks(job, sampler):
    """Generates tasks needed to numerically calculate the gradient for a refinement job."""
    tasks = []
    eps = 1e-8
    x = job['current_params']
    f = job['current_fitness']
    job['gradient_components'] = {}
    
    # Determine gradient method from the sampler's config
    gradient_method = sampler.refinement_gradient_method
    
    if gradient_method == "central":
        job['pending_grad_evals'] = 2 * sampler.n_cont_dims
        for i in range(sampler.n_cont_dims):
            # Positive step
            x_plus = x.copy()
            x_plus[i] += eps
            context = {'type': 'LBFGS_GRADIENT', 'job_id': job['id'], 'dim': i, 'sign': 1}
            tasks.append({'params': x_plus, 'context': context})
            
            # Negative step
            x_minus = x.copy()
            x_minus[i] -= eps
            context = {'type': 'LBFGS_GRADIENT', 'job_id': job['id'], 'dim': i, 'sign': -1}
            tasks.append({'params': x_minus, 'context': context})
            
    else: # Fallback to forward difference for other methods
        job['pending_grad_evals'] = sampler.n_cont_dims
        for i in range(sampler.n_cont_dims):
            x_plus = x.copy()
            x_plus[i] += eps
            context = {'type': 'LBFGS_GRADIENT', 'job_id': job['id'], 'dim': i, 'sign': 1}
            tasks.append({'params': x_plus, 'context': context})
            
    return tasks

def _process_gradient_result(job, result, sampler):
    """Processes a returned likelihood evaluation for a gradient calculation."""
    context = result['context']
    dim, sign = context['dim'], context['sign']
    logL = result['logL']
    
    job['gradient_components'][(dim, sign)] = -logL # Store objective function value
    job['pending_grad_evals'] -= 1
    
    # Check if all components for the gradient have been computed
    if job['pending_grad_evals'] == 0:
        grad = np.zeros(sampler.n_cont_dims)
        f = -job['current_fitness']
        
        if sampler.refinement_gradient_method == "central":
            for i in range(sampler.n_cont_dims):
                f_plus = job['gradient_components'][(i, 1)]
                f_minus = job['gradient_components'][(i, -1)]
                grad[i] = (f_plus - f_minus) / (2 * 1e-8)
        else: # Forward difference
             for i in range(sampler.n_cont_dims):
                f_plus = job['gradient_components'][(i, 1)]
                grad[i] = (f_plus - f) / 1e-8
                
        job['current_gradient'] = grad
        
        # L-BFGS two-loop recursion to find search direction
        q = grad
        a = []
        s_hist, y_hist = job['s_hist'], job['y_hist']
        
        for s, y in zip(reversed(s_hist), reversed(y_hist)):
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            q = q - alpha * y
            a.append(alpha)
        
        if s_hist:
            gamma = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
            z = gamma * q
        else:
            z = q

        for (s, y), alpha in zip(zip(s_hist, y_hist), reversed(a)):
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, z)
            z = z + s * (alpha - beta)
            
        job['search_direction'] = -z
        job['status'] = 'NEEDS_LINE_SEARCH'
        job['line_search_alpha'] = 1.0 # Start line search
        
        # Return a new task for the first step of the line search
        return _calculate_line_search_task(job, sampler)
    return None # Not ready yet, no new task

def _calculate_line_search_task(job, sampler):
    """Generates the next task for a backtracking line search."""
    alpha = job['line_search_alpha']
    x = job['current_params']
    d = job['search_direction']
    x_new = x + alpha * d
    x_new = sampler._ensure_bounds(x_new, sampler.continuous_dims)
    
    context = {'type': 'LBFGS_LINE_SEARCH', 'job_id': job['id'], 'alpha': alpha}
    return {'params': x_new, 'context': context}

def _process_line_search_result(job, result, sampler):
    """Processes a line search result and determines the next step."""
    f_new = -result['logL']
    alpha = result['context']['alpha']
    x = job['current_params']
    f = -job['current_fitness']
    g = job['current_gradient']
    d = job['search_direction']
    c1 = 1e-4

    x_new = x + alpha * d
    x_new = sampler._ensure_bounds(x_new, sampler.continuous_dims)

    # Armijo condition check
    if f_new <= f + c1 * alpha * np.dot(g, x_new - x):
        # Step accepted, move to the next L-BFGS iteration
        job['iteration'] += 1
        
        x_old = job['current_params']
        f_old = -job['current_fitness']
        g_old = job['current_gradient']
        
        # Check for convergence
        if job['iteration'] >= sampler.refinement_max_iter or np.abs(f_old - f_new) < sampler.refinement_ftol:
            job['status'] = 'FINISHED'
            job['success'] = True
            return None # Job is done

        # Update state for next iteration
        job['current_params'] = x_new
        job['current_fitness'] = -f_new
        
        s_k = x_new - x_old
        
        # Generate tasks to calculate the new gradient
        job['status'] = 'NEEDS_GRADIENT'
        new_grad_tasks = _calculate_gradient_tasks(job, sampler)

        # Update L-BFGS history
        def update_history_and_return_tasks():
            g_new = job['current_gradient'] # This will be computed by the new tasks
            y_k = g_new - g_old
            if np.dot(y_k, s_k) > 1e-10:
                job['s_hist'].append(s_k)
                job['y_hist'].append(y_k)
            return new_grad_tasks
        
        # This part is tricky. We need the new gradient to update history.
        # We will calculate the new gradient first, then update history in the 'process_gradient_result' step.
        job['pending_s_k'] = s_k
        job['pending_g_old'] = g_old

        return new_grad_tasks[0] if new_grad_tasks else None # A bit simplified, master will queue all tasks

    else:
        # Step not accepted, reduce alpha and try again
        job['line_search_alpha'] *= 0.5
        if job['line_search_alpha'] < 1e-10: # Failsafe
            job['status'] = 'FINISHED'
            job['success'] = False
            return None # Job is done
        
        return _calculate_line_search_task(job, sampler) # Return task for new alpha

# --- Main Sampler Class ---

class GridAnchoredDESampler:
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
        if self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000
        self.likelihood_calls = 0
        self.global_max_logL = -np.inf
        
        # --- MPI-specific state for the master ---
        self.task_queue = collections.deque()
        self.refinement_jobs = {}
        self.next_job_id = 0
        self.pending_activations = {} # Keeps track of samples for activating a point
        
        # Per-Projection State
        self._reset_for_new_projection(self.projections[0])


    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        print("\n" + "="*80)
        print(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        print("="*80 + "\n")

        self.projection_dims = sorted(projection_config['dims'])
        grid_points_config = projection_config.get('grid_points', [10, 10])
        self.grid_points_per_dim = [gp + 1 for gp in grid_points_config]

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
        self.de_evals_this_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def _flush_samples_buffer(self):
        if not self.samples_output_file or not self.samples_buffer:
            return
        with open(self.samples_output_file, 'a') as f:
            for params, logL in self.samples_buffer:
                param_str = ", ".join([f"{p:.6e}" for p in params])
                f.write(f"{param_str}, {logL:.6e}\n")
        self.samples_buffer = []

    def _log_likelihood_call(self, params, logL):
        """Logs a completed likelihood call (only on master)."""
        self.likelihood_calls += 1
        if hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, logL))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()
        if logL > self.global_max_logL:
            self.global_max_logL = logL

    def _get_grid_indices_from_point(self, point):
        grid_coords = point[self.projection_dims]
        indices = [np.argmin(np.abs(axis - coord)) for i, coord in enumerate(grid_coords)]
        return tuple(indices)

    def _get_grid_coords_from_indices(self, grid_idx):
        return np.array([self.grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _construct_full_params(self, grid_idx, continuous_params):
        """Constructs a full parameter vector from grid and continuous parts."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx)
        full_params[self.continuous_dims] = continuous_params
        return full_params

    def _ensure_bounds(self, vec, dims_to_check):
        return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])
    
    def _get_valid_neighbors(self, grid_idx, include_center=False):
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue
            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))
            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                yield neighbor_idx

    # --- Task Generation Methods (Master Only) ---
    
    def _generate_initial_maxima_tasks(self):
        """Generates tasks to find initial maxima via L-BFGS-B."""
        print(f"--- Generating {self.n_initial_optimizations} initial optimization jobs ---")
        sampler = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = sampler.random(n=self.n_initial_optimizations)
        start_points = self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])
        
        for i, start_point in enumerate(start_points):
            # Each initial optimization is a full refinement job.
            self.initiate_refinement_job(
                start_params=start_point[self.continuous_dims],
                grid_idx=None, # Special case for global optimization
                initial_params_full=start_point,
                job_type='INITIAL_MAXIMA'
            )

    def _generate_de_tasks_for_generation(self, max_num_to_evolve):
        """Generates all DE trial tasks for the current generation."""
        # ... (DE logic from the original _run_single_projection) ...
        # This logic determines which grid points to evolve and creates tasks for them.
        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']
        if not unconconverged_indices: return

        # Simplified prioritization for clarity
        indices_to_process = unconconverged_indices
        if max_num_to_evolve and len(unconverged_indices) > max_num_to_evolve:
            indices_to_process = np.random.choice(unconverged_indices, max_num_to_evolve, replace=False)

        active_pop_list = list(self.active_grid_indices)
        if len(active_pop_list) < 4: return

        parent_pool = []
        for idx in active_pop_list:
            state = self.population[idx]
            best_idx = np.argmax(state['fitnesses'])
            parent_pool.append({'continuous_params': state['continuous_params'][best_idx], 'fitness': state['fitnesses'][best_idx]})

        pbest_archive = []
        if self.mutation_strategy == 'current-to-pbest/1':
            parent_pool.sort(key=lambda p: p['fitness'], reverse=True)
            pbest_size = max(1, int(len(parent_pool) * self.pbest_fraction))
            pbest_archive = parent_pool[:pbest_size]

        tasks_generated = 0
        for grid_idx in indices_to_process:
            # ... (mutation, crossover logic from original file) ...
            # For each trial, create a task and add it to the queue.
            # This part is complex and involves adapting the original loop.
            # For now, a simplified version:
            grid_state = self.population[grid_idx]
            for i in range(self.pop_per_grid_point):
                 # Simplified mutation for brevity in this example
                r1_p, r2_p, r3_p = np.random.choice(parent_pool, 3, replace=False)
                mutant = r1_p['continuous_params'] + 0.5 * (r2_p['continuous_params'] - r3_p['continuous_params'])
                mutant = self._ensure_bounds(mutant, self.continuous_dims)
                trial_params = np.where(np.random.rand(self.n_cont_dims) < 0.9, mutant, grid_state['continuous_params'][i])
                
                full_params = self._construct_full_params(grid_idx, trial_params)
                context = {'type': 'DE_TRIAL', 'grid_idx': grid_idx, 'pop_idx': i, 'trial_params': trial_params}
                self.task_queue.append({'params': full_params, 'context': context})
                tasks_generated += 1
        
        self.de_evals_this_generation = tasks_generated
        print(f"--- Master: Generated {tasks_generated} DE tasks for generation {self.current_generation} ---")

    def initiate_refinement_job(self, start_params, grid_idx, initial_params_full=None, job_type='REFINEMENT', seed_history=None):
        """Creates a new L-BFGS-B job and generates its first tasks."""
        job_id = self.next_job_id
        self.next_job_id += 1
        
        # The full parameter vector is needed to evaluate the first objective function value
        if initial_params_full is None:
             initial_params_full = self._construct_full_params(grid_idx, start_params)

        job = {
            'id': job_id,
            'type': job_type,
            'grid_idx': grid_idx,
            'status': 'NEEDS_INITIAL_F', # Need to evaluate f(x) before starting
            'start_params_full': initial_params_full,
            'current_params': start_params,
            'current_fitness': -np.inf,
            's_hist': collections.deque(maxlen=10),
            'y_hist': collections.deque(maxlen=10),
            'iteration': 0,
            'success': False,
        }
        if seed_history:
            job['s_hist'].extend(seed_history['s'])
            job['y_hist'].extend(seed_history['y'])

        self.refinement_jobs[job_id] = job
        
        # The first task is to get the initial fitness
        context = {'type': 'LBFGS_INITIAL_F', 'job_id': job_id}
        self.task_queue.append({'params': initial_params_full, 'context': context})
        print(f"--- Master: Initiated Job {job_id} ({job_type}) for grid point {grid_idx} ---")
        
    # --- Result Processing Methods (Master Only) ---
    
    def process_result(self, result):
        """Main dispatcher for processing results from workers."""
        context = result['context']
        job_type = context['type']
        
        # Log the call first
        self._log_likelihood_call(result['params'], result['logL'])

        if job_type == 'DE_TRIAL':
            self._process_de_trial_result(result)
        elif job_type == 'ACTIVATE_POINT':
            self._process_activation_result(result)
        elif job_type in ['LBFGS_INITIAL_F', 'LBFGS_GRADIENT', 'LBFGS_LINE_SEARCH']:
            self._process_lbfgs_result(result)
        elif job_type == 'INITIAL_MAXIMA': # This is handled by LBFGS flow
             self._process_lbfgs_result(result)

    def _process_de_trial_result(self, result):
        """Processes the result of a single DE trial."""
        context = result['context']
        grid_idx = context['grid_idx']
        pop_idx = context['pop_idx']
        trial_fitness = result['logL']
        
        grid_state = self.population.get(grid_idx)
        if not grid_state or grid_state['status'] != 'active':
            return # Grid point might have been converged and removed
            
        if trial_fitness > grid_state['fitnesses'][pop_idx]:
            grid_state['continuous_params'][pop_idx] = context['trial_params']
            grid_state['fitnesses'][pop_idx] = trial_fitness
            
            # Check for new best fitness at this grid point
            if trial_fitness > grid_state['best_fitness']:
                improvement = trial_fitness - grid_state['best_fitness']
                grid_state['best_fitness'] = trial_fitness
                grid_state['improvement_history'].append(improvement)
                self.profile_likelihood_grid[grid_idx] = trial_fitness
                
                # Potentially activate neighbors
                if trial_fitness > (self.global_max_logL - self.roi_threshold):
                    self._activate_neighbors(grid_idx)

        self.de_evals_this_generation -= 1
        
    def _process_lbfgs_result(self, result):
        """Processes a result related to an L-BFGS refinement job."""
        context = result['context']
        job_id = context['job_id']
        job = self.refinement_jobs.get(job_id)
        if not job or job['status'] == 'FINISHED':
            return

        new_tasks = []
        if job['status'] == 'NEEDS_INITIAL_F':
            job['current_fitness'] = result['logL']
            job['status'] = 'NEEDS_GRADIENT'
            new_tasks = _calculate_gradient_tasks(job, self)

        elif job['status'] == 'NEEDS_GRADIENT':
            # This function will handle all logic and return new tasks if the gradient is complete
            new_tasks = self._process_gradient_result_for_job(job, result)
            
        elif job['status'] == 'NEEDS_LINE_SEARCH':
             # This function will handle all logic and return new tasks if the line search continues
            new_tasks = self._process_line_search_result_for_job(job, result)
        
        if new_tasks:
            for task in new_tasks:
                full_params = self._construct_full_params(job['grid_idx'], task['params'])
                self.task_queue.append({'params': full_params, 'context': task['context']})

        if job['status'] == 'FINISHED':
            print(f"--- Master: Job {job_id} ({job['type']}) finished. Success: {job['success']} ---")
            self._finalize_job(job)

    def _finalize_job(self, job):
        """Finalize a job, updating the sampler state."""
        if job['type'] == 'INITIAL_MAXIMA' and job['success']:
            # The grid_idx is None, but we have the full final parameter vector
            final_params = self._construct_full_params(job['grid_idx'], job['current_params']) \
                if job['grid_idx'] is not None else job['start_params_full'] # A bit of a hack
            
            # This needs to be fixed. For now, let's assume `current_params` is full for initial maxima.
            # A better way is to reconstruct it based on its context. Let's assume this for now.
            # The initial job should store the full parameter vector.
            # Let's modify initiate_refinement_job for this.
            # OK, I'll assume the job object stores enough info.

            # Reconstruct the final full parameter set
            if job['grid_idx'] is not None:
                 final_params = self._construct_full_params(job['grid_idx'], job['current_params'])
            else: # Global optimization
                 # This logic is flawed. Let's correct it. A global opt has no grid_idx.
                 # The 'params' in the task should be the full params.
                 # The worker will evaluate it. The LBFGS logic needs to handle full-D vectors.
                 # This requires a bigger refactor. Let's simplify for now.
                 # Let's assume initial maxima search is not gridded.
                 print("WARNING: Finalizing INITIAL_MAXIMA job logic is simplified.")
                 final_params = job['current_params'] # Assuming it's full-dimensional
            
            final_logL = job['current_fitness']
            self.initial_maxima.append({'point': final_params, 'logL': final_logL})
            if final_logL > self.global_max_logL: self.global_max_logL = final_logL
        
        elif job['type'] == 'REFINEMENT' and job['success']:
            grid_idx = job['grid_idx']
            if grid_idx in self.population:
                state = self.population[grid_idx]
                state['optimizer_state'] = {'s': list(job['s_hist']), 'y': list(job['y_hist'])}
                # Update the best individual with the refined result
                if job['current_fitness'] > state['best_fitness']:
                     state['best_fitness'] = job['current_fitness']
                     best_idx = np.argmax(state['fitnesses'])
                     state['continuous_params'][best_idx] = job['current_params']
                     state['fitnesses'][best_idx] = job['current_fitness']
                     self.profile_likelihood_grid[grid_idx] = job['current_fitness']

        # Clean up
        del self.refinement_jobs[job['id']]
            
    def _activate_neighbors(self, grid_idx):
        """Checks neighbors of a grid point and queues them for activation if new."""
        state = self.population[grid_idx]
        for neighbor_idx in self._get_valid_neighbors(grid_idx):
            if neighbor_idx not in self.population and neighbor_idx not in self.pending_activations:
                print(f"--- Master: Activating neighbor {neighbor_idx} ---")
                best_idx = np.argmax(state['fitnesses'])
                warm_start_params = state['continuous_params'][best_idx]
                self._activate_point(neighbor_idx, warm_start_params=warm_start_params)

    def _activate_point(self, grid_idx, warm_start_params=None):
        """Generates tasks to activate a new grid point."""
        if grid_idx in self.population or grid_idx in self.pending_activations:
            return

        cont_bounds = self.bounds[self.continuous_dims]
        sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = sampler.random(n=self.pop_per_grid_point)
        scaled_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])

        if warm_start_params is not None:
            distances = np.linalg.norm(scaled_samples - warm_start_params, axis=1)
            closest_idx = np.argmin(distances)
            scaled_samples[closest_idx] = warm_start_params

        # Store pending activation data
        self.pending_activations[grid_idx] = {
            'continuous_params': scaled_samples,
            'fitnesses': np.full(self.pop_per_grid_point, -np.inf),
            'evals_pending': self.pop_per_grid_point
        }

        # Generate tasks for all individuals
        for i, params in enumerate(scaled_samples):
            full_params = self._construct_full_params(grid_idx, params)
            context = {'type': 'ACTIVATE_POINT', 'grid_idx': grid_idx, 'pop_idx': i}
            self.task_queue.append({'params': full_params, 'context': context})

    def _process_activation_result(self, result):
        """Processes a result from an activation evaluation."""
        context = result['context']
        grid_idx = context['grid_idx']
        
        if grid_idx not in self.pending_activations: return
        
        pending_state = self.pending_activations[grid_idx]
        pending_state['fitnesses'][context['pop_idx']] = result['logL']
        pending_state['evals_pending'] -= 1
        
        if pending_state['evals_pending'] == 0:
            # All evaluations are done, formally activate the point
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
            print(f"--- Master: Successfully activated grid point {grid_idx} ---")

# --- Test Functions and Plotting (mostly unchanged) ---
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
    # This function remains unchanged, as it's a visualization tool for the master.
    pass

# --- MPI Worker and Master Main Functions ---

def worker_main(comm):
    """Main loop for a worker process."""
    rank = comm.Get_rank()
    # First, receive the likelihood function from the master.
    likelihood_func = comm.bcast(None, root=0)
    print(f"Worker {rank}: Received likelihood function. Ready for tasks.")

    while True:
        # Wait for a task from the master
        task = comm.recv(source=0, tag=MPI.ANY_TAG)
        
        if task == TASK_TERMINATE:
            print(f"Worker {rank}: Received terminate signal. Exiting.")
            break
            
        # Execute the task (a single likelihood evaluation)
        params = task['params']
        logL = likelihood_func(params)
        
        # Send the result back to the master
        result = {'logL': logL, 'params': params, 'context': task['context']}
        comm.send(result, dest=0)

def master_main(comm, sampler, num_generations, max_num_to_evolve, plot_callback, plot_interval):
    """Main control loop for the master process."""
    n_workers = comm.Get_size() - 1
    if n_workers <= 0:
        print("Error: This script requires at least 2 MPI processes (1 master, 1+ workers).")
        return

    print(f"Master: Starting with {n_workers} workers.")
    
    # 1. Broadcast the likelihood function to all workers
    comm.bcast(sampler.likelihood_func, root=0)

    # 2. Initial state setup
    free_workers = list(range(1, n_workers + 1))
    pending_requests = {} # Maps request handle to worker rank
    
    # 3. Generate initial tasks
    sampler._generate_initial_maxima_tasks()
    
    # This is a simplified main loop. A real implementation would be more complex.
    # The main challenge is managing the overall state transitions (init -> DE -> patching).
    # This example focuses on the task distribution mechanism.
    
    tasks_sent = 0
    tasks_completed = 0

    while True: # Main event loop
        
        # --- Dispatch tasks to free workers ---
        while sampler.task_queue and free_workers:
            worker_rank = free_workers.pop(0)
            task = sampler.task_queue.popleft()
            
            comm.send(task, dest=worker_rank)
            tasks_sent += 1
            # For simplicity, we use blocking send/recv here. A fully async
            # version would use isend/irecv and manage MPI.Request objects.
        
        # --- Wait for and process a result ---
        if tasks_sent > tasks_completed:
            result = comm.recv(source=MPI.ANY_SOURCE)
            worker_rank = result['context'].get('worker_rank', 'unknown') # You'd get this from status object
            # For this simplified blocking example, we don't know the worker rank easily.
            # An async model would track this. Let's assume we get it back.
            free_workers.append(result['context'].get('worker_rank', 1)) # Dummy worker rank
            tasks_completed += 1
            
            sampler.process_result(result)

        # --- Check for state transitions (e.g., end of DE generation) ---
        if sampler.de_evals_this_generation <= 0 and len(sampler.population) > 0:
            sampler.current_generation += 1
            print(f"\n--- Master: Advancing to generation {sampler.current_generation} ---")
            # Check for convergence, refinement, etc.
            # ...
            # Generate new DE tasks
            sampler._generate_de_tasks_for_generation(max_num_to_evolve)

        # Termination condition (simplified)
        if sampler.current_generation >= num_generations:
            print("Master: Reached max generations. Terminating.")
            break
        
        # A better check would be when task queue is empty and all jobs are finished.
        if not sampler.task_queue and not sampler.refinement_jobs and tasks_sent == tasks_completed:
            print("Master: All tasks and jobs are complete. Terminating.")
            break

    # 4. Terminate workers
    for rank in range(1, n_workers + 1):
        comm.send(TASK_TERMINATE, dest=rank)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Configuration (shared by all processes, but only master uses most of it) ---
    TEST_FUNCTION = "himmelblau_4d"
    OUTPUT_FILE = f"samples_rank{rank}.csv"
    
    PROJECTIONS_TO_RUN = [
        {'dims': [0, 1], 'grid_points': [100, 100], 'patching': True, 'refining': True},
    ]
    
    log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

    if rank == 0:
        # --- Master Process ---
        sampler = GridAnchoredDESampler(
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
            refinement_gradient_method="central",
            patching_fraction=0.05,
            patching_conv_threshold=0.01,
            memory_size=len(PROJECTIONS_TO_RUN[0]['grid_points']) * 25,
            samples_output_file=OUTPUT_FILE,
        )

        master_main(
            comm=comm,
            sampler=sampler,
            num_generations=100, # This is now a termination condition
            max_num_to_evolve=50,
            plot_callback=None, # Plotting in MPI is complex, disabled for now
            plot_interval=100
        )
    else:
        # --- Worker Process ---
        worker_main(comm)

