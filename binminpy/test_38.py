import numpy as np
import itertools
import collections
import time
from scipy.optimize import minimize
from scipy.stats import cauchy, norm
from scipy.stats.qmc import LatinHypercube as LHS
import os


# --- Custom L-BFGS-B Optimizer ---

class CustomLBFGSB:
    """
    A custom implementation of the L-BFGS-B algorithm.
    This is tailored for the sampler to allow for Hessian seeding.
    """
    def __init__(self, objective_func, bounds, ftol=1e-9, gtol=1e-7, max_iter=50, history_size=10, gradient_method="central"):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.ftol = ftol
        self.gtol = gtol
        self.max_iter = max_iter
        self.history_size = history_size
        self.dim = len(self.bounds)
        self.gradient_method = gradient_method

    def _calculate_gradient(self, x, f):
        """Numerically calculates the gradient."""
        grad = np.zeros_like(x)
        eps = 1e-8
        if self.gradient_method == "forward":
            for i in range(self.dim):
                x_plus = x.copy()
                x_plus[i] += eps
                grad[i] = (self.objective_func(x_plus) - f) / eps
        elif self.gradient_method == "backward":
            for i in range(self.dim):
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (f - self.objective_func(x_minus)) / eps
        elif self.gradient_method == "forward-backward-alternate":
            current_method = np.random.choice(["f", "b"])
            for i in range(self.dim):
                if current_method == "f": 
                    x_plus = x.copy()
                    x_plus[i] += eps
                    grad[i] = (self.objective_func(x_plus) - f) / eps
                    current_method = "b"
                elif current_method == "b":
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    grad[i] = (f - self.objective_func(x_minus)) / eps
                    current_method = "f"
        elif self.gradient_method == "central":
            for i in range(self.dim):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (self.objective_func(x_plus) - self.objective_func(x_minus)) / (2 * eps)
        return grad

    def _line_search(self, x, f, g, d, alpha_init=1.0, c1=1e-4, rho=0.5):
        """Backtracking line search satisfying the Armijo condition."""
        alpha = alpha_init
        while True:
            x_new = x + alpha * d
            # Project back into bounds
            x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
            f_new = self.objective_func(x_new)
            if f_new <= f + c1 * alpha * np.dot(g, x_new - x):
                return x_new, f_new, alpha
            alpha *= rho
            if alpha < 1e-10: # Failsafe
                return x, f, 0

    def run(self, x0, initial_history=None):
        """Executes the optimization loop."""
        x = np.clip(x0, self.bounds[:, 0], self.bounds[:, 1])
        f = self.objective_func(x)
        g = self._calculate_gradient(x,f)

        s_hist = collections.deque(maxlen=self.history_size)
        y_hist = collections.deque(maxlen=self.history_size)

        if initial_history:
            s_hist.extend(initial_history['s'])
            y_hist.extend(initial_history['y'])

        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.gtol:
                break

            # Two-loop recursion to find search direction
            q = g
            a = []
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
            
            d = -z
            
            x_old, f_old, g_old = x, f, g
            x, f, step = self._line_search(x_old, f_old, g_old, d)

            if step == 0 or np.abs(f_old - f) < self.ftol:
                break

            g = self._calculate_gradient(x,f)
            
            s_k = x - x_old
            y_k = g - g_old

            if np.dot(y_k, s_k) > 1e-10: # Ensure curvature condition
                s_hist.append(s_k)
                y_hist.append(y_k)

        final_state = {'s': list(s_hist), 'y': list(y_hist)}
        return {'params': x, 'fitness': -f, 'success': True, 'optimizer_state': final_state}

# --- Main Sampler Class ---

class GridAnchoredDESampler:
    """
    Implements a Grid-Anchored Differential Evolution sampler.
    ... (docstring unchanged) ...
    """

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
        """
        Initializes the Grid-Anchored DE Sampler.
        ... (args docstring updated) ...
        """
        self.likelihood_func = likelihood_func
        self.likelihood_func_name = likelihood_func_name
        self.bounds = np.array(bounds)
        self.dims = len(self.bounds)

        # --- Projection setup ---
        if not isinstance(projections, list):
            raise TypeError("projections must be a list of dictionaries.")
        self.projections = projections

        # --- Algorithm parameters ---
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
        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        if self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000

        # --- Persistent State (across projections) ---
        self.likelihood_calls = 0
        self.global_max_logL = -np.inf
        
        # --- Per-Projection State (will be reset) ---
        self.projection_dims = None
        self.grid_points_per_dim = None
        self.initial_maxima = []
        self.population = {}
        self.active_grid_indices = set()
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0


    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        print("\n" + "="*80)
        print(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        print("="*80 + "\n")

        self.projection_dims = sorted(projection_config['dims'])
        # _Anders: Add +1 to number of grid points, to get nicer coordinates
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

        # Reset state variables
        self.initial_maxima = []
        self.population = {}
        self.active_grid_indices = set()
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def _flush_samples_buffer(self):
        """Writes the content of the samples buffer to the output file."""
        if not self.samples_output_file or not self.samples_buffer:
            return
        
        with open(self.samples_output_file, 'a') as f:
            for params, logL in self.samples_buffer:
                param_str = ", ".join([f"{p:.6e}" for p in params])
                f.write(f"{param_str}, {logL:.6e}\n")
        
        self.samples_buffer = []

    def _call_and_log_likelihood(self, params):
        """Central wrapper for calling the likelihood function."""
        logL = self.likelihood_func(params)
        self.likelihood_calls += 1

        if hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, logL))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()
        
        return logL

    def _get_grid_indices_from_point(self, point, grid_axes=None):
        """Converts a point's projection coordinates to the closest grid indices."""
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
        """Converts grid indices to projection parameter values."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _evaluate_likelihood(self, grid_idx, continuous_params, grid_axes_override=None):
        """Constructs a full parameter vector and evaluates the likelihood via the central logger."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes_override)
        full_params[self.continuous_dims] = continuous_params
        return self._call_and_log_likelihood(full_params)

    def _ensure_bounds(self, vec, dims_to_check):
        """Ensures a vector's components are within the defined bounds."""
        return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])
    
    def _get_valid_neighbors(self, grid_idx, include_center=False):
        """Generator to yield valid neighbor indices for a given grid point."""
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue
            
            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))
            
            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                yield neighbor_idx

    def _find_initial_maxima(self):
        """Runs L-BFGS-B from random start points to find initial maxima."""
        print(f"--- Running {self.n_initial_optimizations} initial optimizations to find maxima ---")
        
        def objective_func(params):
            return -self._call_and_log_likelihood(params)

        # Generate start points using Latin Hypercube Sampling for better coverage
        sampler = LHS(d=self.dims, seed=np.random.randint(1000000, 1000000000000))
        unit_samples = sampler.random(n=self.n_initial_optimizations)
        start_points = self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])

        for i, start_point in enumerate(start_points):
            res = minimize(objective_func, start_point, method='L-BFGS-B', bounds=self.bounds)
            
            if res.success:
                logL = -res.fun
                point = res.x
                self.initial_maxima.append({'point': point, 'logL': logL})
                if logL > self.global_max_logL:
                    self.global_max_logL = logL
                print(f"  Start {i+1}/{self.n_initial_optimizations}: Found maximum with logL = {logL:.4e}")
        
        if not self.initial_maxima:
            print("--- Initial optimization failed to find any valid maxima. ---")
            return

        self.initial_maxima.sort(key=lambda x: x['logL'], reverse=True)
        print(f"--- Found {len(self.initial_maxima)} maxima. Global max logL = {self.global_max_logL:.4e} ---")

    def _initialize_from_warm_start_file(self, warm_start_file):
        """Initializes grid points from a previous sample file."""
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
        
        activated_count = 0
        for grid_idx, candidate in best_candidates.items():
            if candidate['logL'] < roi_cutoff:
                continue
            if grid_idx in self.population:
                continue
            
            # Initialize population with NumPy arrays
            continuous_params = np.zeros((self.pop_per_grid_point, self.n_cont_dims))
            fitnesses = np.full(self.pop_per_grid_point, -np.inf)
            
            warm_start_params = candidate['params'][self.continuous_dims]
            cont_bounds = self.bounds[self.continuous_dims]

            sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1000000, 1000000000000))
            unit_samples = sampler.random(n=self.pop_per_grid_point)
            scaled_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])

            distances = np.linalg.norm(scaled_samples - warm_start_params, axis=1)
            closest_idx = np.argmin(distances)
            scaled_samples[closest_idx] = warm_start_params

            for i, params in enumerate(scaled_samples):
                continuous_params[i] = params
                if i == closest_idx:
                    fitnesses[i] = candidate['logL']
                else:
                    fitnesses[i] = self._evaluate_likelihood(grid_idx, params)
            
            best_fitness = np.max(fitnesses)
            self.profile_likelihood_grid[grid_idx] = best_fitness
            
            self.population[grid_idx] = {
                'continuous_params': continuous_params,
                'fitnesses': fitnesses,
                'best_fitness': best_fitness,
                'status': 'active',
                'improvement_history': collections.deque(maxlen=self.convergence_window),
                'last_update_gen': 0,
                'optimizer_state': None
            }
            self.active_grid_indices.add(grid_idx)
            activated_count += 1

        print(f"--- Activated {activated_count} grid points from file. New Global Max logL: {self.global_max_logL:.4e} ---")

    def _activate_point(self, grid_idx, warm_start_params=None):
        """Activates a new grid point for evolution using Latin Hypercube Sampling."""
        if grid_idx in self.population:
            return

        continuous_params = np.zeros((self.pop_per_grid_point, self.n_cont_dims))
        fitnesses = np.full(self.pop_per_grid_point, -np.inf)
        cont_bounds = self.bounds[self.continuous_dims]
        
        sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1000000, 1000000000000))
        unit_samples = sampler.random(n=self.pop_per_grid_point)
        scaled_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])

        if warm_start_params is not None:
            distances = np.linalg.norm(scaled_samples - warm_start_params, axis=1)
            closest_idx = np.argmin(distances)
            scaled_samples[closest_idx] = warm_start_params

        for i, params in enumerate(scaled_samples):
            continuous_params[i] = params
            fitnesses[i] = self._evaluate_likelihood(grid_idx, params)

        best_fitness = np.max(fitnesses)
        self.profile_likelihood_grid[grid_idx] = best_fitness

        self.population[grid_idx] = {
            'continuous_params': continuous_params,
            'fitnesses': fitnesses,
            'best_fitness': best_fitness,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=self.convergence_window),
            'last_update_gen': 0,
            'optimizer_state': None
        }
        self.active_grid_indices.add(grid_idx)

    def _initialize_population_from_maxima(self):
        """Activates grid points around the pre-found maxima."""
        if not self.initial_maxima:
            return
        print("--- Initializing population around found maxima ---")
        for maximum in self.initial_maxima:
            point = maximum['point']
            grid_idx = self._get_grid_indices_from_point(point)
            
            for neighbor_idx in self._get_valid_neighbors(grid_idx, include_center=True):
                self._activate_point(neighbor_idx, warm_start_params=point[self.continuous_dims])
        print(f"--- Total active grid points after maxima init: {len(self.population)} ---")


    def _refine_single_point(self, grid_idx, use_neighbor_info=True):
        """
        Refines a single grid point using the custom L-BFGS-B optimizer
        with Hessian seeding from the best converged neighbor.
        """
        state = self.population[grid_idx]
        
        # --- 1. Find the best converged neighbor ---
        best_neighbor_state = None
        best_neighbor_fitness = -np.inf
        if use_neighbor_info:
            for neighbor_idx in self._get_valid_neighbors(grid_idx):
                if neighbor_idx in self.population:
                    neighbor_state = self.population[neighbor_idx]
                    if neighbor_state['status'] == 'converged' and neighbor_state.get('optimizer_state') is not None:
                        if neighbor_state['best_fitness'] > best_neighbor_fitness:
                            best_neighbor_fitness = neighbor_state['best_fitness']
                            best_neighbor_state = neighbor_state

        # --- 2. Determine the best starting point ---
        current_best_idx = np.argmax(state['fitnesses'])
        start_params = state['continuous_params'][current_best_idx].copy()
        current_best_fitness = state['fitnesses'][current_best_idx]

        if best_neighbor_state:
            neighbor_best_idx = np.argmax(best_neighbor_state['fitnesses'])
            neighbor_params = best_neighbor_state['continuous_params'][neighbor_best_idx]
            
            # Test neighbor's params at the current grid location
            neighbor_test_fitness = self._evaluate_likelihood(grid_idx, neighbor_params)
            if neighbor_test_fitness > current_best_fitness:
                start_params = neighbor_params
                current_best_fitness = neighbor_test_fitness
                # Update individual to reflect this better starting point
                state['continuous_params'][current_best_idx] = start_params
                state['fitnesses'][current_best_idx] = current_best_fitness
                state['best_fitness'] = np.max(state['fitnesses'])
                self.profile_likelihood_grid[grid_idx] = state['best_fitness']
        
        # --- 3. Initialize and run the custom optimizer ---
        def objective_func(p):
            return -self._evaluate_likelihood(grid_idx, p)

        optimizer = CustomLBFGSB(
            objective_func,
            bounds=self.bounds[self.continuous_dims],
            ftol=self.refinement_ftol,
            gtol=1e-7,
            max_iter=self.refinement_max_iter,
            gradient_method=self.refinement_gradient_method
        )
        
        initial_history = best_neighbor_state['optimizer_state'] if best_neighbor_state else None
        
        result = optimizer.run(start_params, initial_history=initial_history)

        # --- 4. Update state with the result ---
        improvement = 0.0
        if result['success'] and result['fitness'] > current_best_fitness:
            improvement = result['fitness'] - current_best_fitness
            
            # Find best individual again in case it changed
            best_idx = np.argmax(state['fitnesses'])
            state['continuous_params'][best_idx] = result['params']
            state['fitnesses'][best_idx] = result['fitness']
            
            new_best_fitness = np.max(state['fitnesses'])
            state['best_fitness'] = new_best_fitness
            self.profile_likelihood_grid[grid_idx] = new_best_fitness
            
            if new_best_fitness > self.global_max_logL:
                self.global_max_logL = new_best_fitness
            
        # Store the state even if it didn't improve, for future seeding
        state['optimizer_state'] = result['optimizer_state']
        return improvement

    def _run_final_patching(self, plot_callback=None, fig=None, axes=None, points_to_skip=None):
        """
        Runs a single patching pass to refine points with high positive gradients
        relative to their neighbors within the ROI, prioritized by likelihood.
        """
        print("\n--- Starting Final Patching Pass ---")

        # 1. Identify all evaluated points within the ROI.
        roi_cutoff = self.global_max_logL - self.roi_threshold
        candidate_indices = []
        for grid_idx, state in self.population.items():

            if points_to_skip and points_to_skip.count(grid_idx) > 2:
                continue

            if state['best_fitness'] >= roi_cutoff:
                candidate_indices.append(grid_idx)
            else:
                neighbor_count = 0
                roi_neighbor_count = 0
                for neighbor_idx in self._get_valid_neighbors(grid_idx):
                    neighbor_count += 1
                    if self.profile_likelihood_grid[neighbor_idx] >= roi_cutoff:
                        roi_neighbor_count += 1
                if neighbor_count > 0 and (roi_neighbor_count / neighbor_count) > 0.5:
                    candidate_indices.append(grid_idx)

        if not candidate_indices:
            print("  No evaluated points in ROI to consider for patching. Skipping.")
            return 0.0, [], []

        priority_scores = []

        # 2. Calculate priority scores for each candidate point.
        for grid_idx in candidate_indices:

            current_logL = self.profile_likelihood_grid[grid_idx]
            scalar_gradient_sum = 0.0
            neighbor_logL_sum = 0.0

            # Iterate over all dimensions and both directions (+1 and -1)
            for dim_idx in range(self.n_proj_dims):
                for direction in [-1, 1]:
                    offset = np.zeros(self.n_proj_dims, dtype=int)
                    offset[dim_idx] = direction
                    neighbor_idx = tuple(np.array(grid_idx) + offset)

                    if not all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                        continue

                    neighbor_logL = self.profile_likelihood_grid[neighbor_idx]

                    if neighbor_logL > -np.inf and neighbor_logL >= roi_cutoff:
                        gradient = neighbor_logL - current_logL
                        if gradient > 0:
                            scalar_gradient_sum += gradient
                            neighbor_logL_sum += neighbor_logL

            if scalar_gradient_sum > 0:
                likelihood_weight = max(neighbor_logL_sum - (self.global_max_logL - self.roi_threshold), 0) + 1.0
                priority_score = scalar_gradient_sum * likelihood_weight
                priority_scores.append((priority_score, grid_idx))

        if not priority_scores:
            print("  No points found with positive outward gradients within the ROI. Patching complete.")
            return 0.0, [], []

        # 3. Sort points by their priority score in descending order.
        priority_scores.sort(key=lambda x: x[0], reverse=True)

        # 4. Determine the number of points to patch based on the fraction.
        num_to_patch = int(len(priority_scores) * self.patching_fraction)
        if num_to_patch == 0 and len(priority_scores) > 0:
            num_to_patch = 1 # Ensure at least one point is patched if possible

        points_to_patch = priority_scores[:num_to_patch]

        print(f"  Identified {len(points_to_patch)} candidates for patching (top {self.patching_fraction*100:.1f}%).")

        # Visualize the points selected for patching
        if plot_callback and fig and np.all(axes):
            patch_coords = np.array([self._get_grid_coords_from_indices(idx) for _, idx in points_to_patch])
            if patch_coords.size > 0:
                patch_plot = axes[0].scatter(patch_coords[:, 0], patch_coords[:, 1], c='red', s=20, marker='x', label='Patch Candidates', zorder=10)
                if any(l is not None for l in axes[0].get_legend_handles_labels()):
                    axes[0].legend()
                fig.canvas.draw()
                import matplotlib.pyplot as plt
                plt.pause(0.01)
                patch_plot.remove()

        # 5. Execute patching.
        tot_improvement_this_iter = 0.0
        n_improvements_this_iter = 0
        points_attempted_patched = []
        points_successfully_patched = []
        for _, grid_idx in points_to_patch:
            points_attempted_patched.append(grid_idx)
            point_improvement = self._refine_single_point(grid_idx, use_neighbor_info=True)
            if point_improvement > 0:
                points_successfully_patched.append(grid_idx)
                n_improvements_this_iter += 1
                tot_improvement_this_iter += point_improvement

        print(f"Patching | Total Improvement: {tot_improvement_this_iter} | Improved points: {n_improvements_this_iter} | "
              f"New Max logL: {self.global_max_logL:.4e}")

        return tot_improvement_this_iter, points_attempted_patched, points_successfully_patched


    def run_projections(self, num_generations, max_num_to_evolve=None, plot_callback=None, plot_interval=10, skip_init_opt_on_warm_start=True):
        """Main entry point to run a sequence of projections."""
        fig, axes = None, None
        if plot_callback:
            try:
                import matplotlib.pyplot as plt
                plt.ion()
                fig, axes = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [20, 1]})
            except ImportError:
                print("\nMatplotlib not found. Skipping visualization.")
                plot_callback = None

        for i, proj_config in enumerate(self.projections):
            self._reset_for_new_projection(proj_config)
            
            warm_start_file = self.samples_output_file
            skip_initial_opt = skip_init_opt_on_warm_start and (i > 0)

            self._run_single_projection(
                num_generations=num_generations,
                max_num_to_evolve=max_num_to_evolve,
                plot_callback=plot_callback,
                plot_interval=plot_interval,
                warm_start_file=warm_start_file,
                skip_initial_optimizations_on_warm_start=skip_initial_opt,
                projection_config=proj_config,
                fig=fig,
                axes=axes
            )

            if plot_callback:
                proj_dims_str = '_'.join(map(str, proj_config['dims']))
                filename = f"plot__{self.likelihood_func_name}__{proj_dims_str}.png"
                print(f"--- Saving final plot to {filename} ---")
                fig.savefig(filename, dpi=300)

            input("Press a key to continue...")

        if hasattr(self, 'samples_buffer'):
            print(f"\nFlushing remaining {len(self.samples_buffer)} samples to {self.samples_output_file}...")
            self._flush_samples_buffer()
        
        if plot_callback:
            print("\nRun finished. Close the plot to exit.")
            plt.ioff()
            plt.show()

        print("\n--- All Projections Complete ---")

    def _run_single_projection(self, num_generations, max_num_to_evolve, plot_callback, plot_interval,
                               warm_start_file, skip_initial_optimizations_on_warm_start,
                               projection_config, fig=None, axes=None):
        """Internal method to run the sampling loop for one projection."""
        if not skip_initial_optimizations_on_warm_start:
            self._find_initial_maxima()
        
        self._initialize_from_warm_start_file(warm_start_file)
        
        self._initialize_population_from_maxima()
        
        if plot_callback:
            plot_callback(self, fig, axes)

        print("\n--- Starting Grid-Anchored DE Evolution ---")
        start_time = time.time()
        self.current_generation = 0
        while self.current_generation < num_generations:
            self.current_generation += 1
            
            successful_F = []
            successful_CR = []
            
            unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']
            
            if not unconverged_indices:
                print("All active points have converged. Ending DE phase.")
                break

            priority_scores = []
            for idx in unconverged_indices:
                state = self.population[idx]
                fitness_score = max(0, state['best_fitness'] - (self.global_max_logL - 2 * self.roi_threshold))
                improvement_rate = np.mean(state['improvement_history']) if state['improvement_history'] else 0
                improvement_score = improvement_rate * 10
                pri_score = fitness_score + improvement_score + (1./len(unconverged_indices))
                priority_scores.append(pri_score)
            
            priority_scores = np.array(priority_scores)
            probabilities = priority_scores / np.sum(priority_scores)

            if max_num_to_evolve is not None:
                num_to_evolve = min(len(unconverged_indices), max_num_to_evolve)
            else:
                num_to_evolve = len(unconverged_indices)
            
            indices_to_process_map = np.random.choice(
                np.arange(len(unconverged_indices)),
                size=num_to_evolve, 
                replace=False, 
                p=probabilities
            )
            indices_to_process = [unconverged_indices[i] for i in indices_to_process_map]

            active_pop_list = list(self.active_grid_indices)
            if len(active_pop_list) < 4:
                continue 

            # Create a temporary pool of the best individuals from each active grid point
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

            for grid_idx in indices_to_process:
                grid_state = self.population[grid_idx]
                old_best_fitness = grid_state['best_fitness']

                for i in range(self.pop_per_grid_point):
                    mem_loc = np.random.randint(0, self.memory_size)
                    mu_CR, mu_F = self.memory_CR[mem_loc], self.memory_F[mem_loc]
                    
                    CR_i = np.clip(norm.rvs(loc=mu_CR, scale=0.1), 0, 1)
                    F_i = cauchy.rvs(loc=mu_F, scale=0.1)
                    while F_i <= 0:
                        F_i = cauchy.rvs(loc=mu_F, scale=0.1)
                    F_i = min(F_i, 1.0)

                    x_i_params = grid_state['continuous_params'][i]
                    
                    use_neighbor_mutation = False
                    best_neighbor_params = None
                    if np.random.rand() < self.neighbor_pull_probability:
                        best_neighbor_fitness = -np.inf
                        for neighbor_idx in self._get_valid_neighbors(grid_idx):
                            if neighbor_idx in self.population:
                                neighbor_state = self.population[neighbor_idx]
                                if neighbor_state['best_fitness'] > best_neighbor_fitness:
                                    best_neighbor_fitness = neighbor_state['best_fitness']
                                    neighbor_best_idx = np.argmax(neighbor_state['fitnesses'])
                                    best_neighbor_params = neighbor_state['continuous_params'][neighbor_best_idx]
                        
                        if best_neighbor_params is not None and best_neighbor_fitness > grid_state['best_fitness']:
                            use_neighbor_mutation = True
                    
                    # The check to prevent using the current individual in mutation is removed for simplicity,
                    # as it was fragile and has a low probability of occurring in a large pool.
                    if len(parent_pool) < 3:
                        continue

                    mutant = None
                    if use_neighbor_mutation:
                        r2_p, r3_p = np.random.choice(parent_pool, 2, replace=False)
                        r2, r3 = r2_p['continuous_params'], r3_p['continuous_params']
                        mutant = x_i_params + F_i * (best_neighbor_params - x_i_params) + F_i * (r2 - r3)
                    
                    elif self.mutation_strategy == 'current-to-rand/1':
                        p1_p, p2_p, p3_p = np.random.choice(parent_pool, 3, replace=False)
                        p1, p2, p3 = p1_p['continuous_params'], p2_p['continuous_params'], p3_p['continuous_params']
                        mutant = x_i_params + F_i * (p1 - x_i_params) + F_i * (p2 - p3)

                    elif self.mutation_strategy == 'rand/1':
                        r1_p, r2_p, r3_p = np.random.choice(parent_pool, 3, replace=False)
                        r1, r2, r3 = r1_p['continuous_params'], r2_p['continuous_params'], r3_p['continuous_params']
                        mutant = r1 + F_i * (r2 - r3)

                    elif self.mutation_strategy == 'current-to-pbest/1':
                        if not pbest_archive:
                            pbest_archive = parent_pool
                        
                        x_pbest_p = np.random.choice(pbest_archive)
                        x_pbest = x_pbest_p['continuous_params']

                        potential_diff = [p for p in parent_pool if p is not x_pbest_p]
                        if len(potential_diff) < 2:
                            continue
                        
                        r2_p, r3_p = np.random.choice(potential_diff, 2, replace=False)
                        r2, r3 = r2_p['continuous_params'], r3_p['continuous_params']
                        mutant = x_i_params + F_i * (x_pbest - x_i_params) + F_i * (r2 - r3)

                    mutant = self._ensure_bounds(mutant, self.continuous_dims)

                    cross_points = np.random.rand(self.n_cont_dims) < CR_i
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.n_cont_dims)] = True
                    trial_params = np.where(cross_points, mutant, x_i_params)

                    trial_fitness = self._evaluate_likelihood(grid_idx, trial_params)
                    
                    if trial_fitness > grid_state['fitnesses'][i]:
                        grid_state['continuous_params'][i] = trial_params
                        grid_state['fitnesses'][i] = trial_fitness
                        successful_F.append(F_i)
                        successful_CR.append(CR_i)

                new_best_fitness = np.max(grid_state['fitnesses'])
                improvement = new_best_fitness - old_best_fitness
                grid_state['improvement_history'].append(improvement)
                
                if new_best_fitness > old_best_fitness:
                    grid_state['best_fitness'] = new_best_fitness
                    grid_state['last_update_gen'] = self.current_generation
                    self.profile_likelihood_grid[grid_idx] = new_best_fitness
                    if new_best_fitness > self.global_max_logL:
                        self.global_max_logL = new_best_fitness

            if successful_F:
                weights = np.ones(len(successful_F))
                muF = np.sum(weights * np.array(successful_F)**2) / np.sum(weights * np.array(successful_F))
                muCR = np.sum(weights * np.array(successful_CR)) / np.sum(weights)
                
                self.memory_F[self.memory_idx] = muF
                self.memory_CR[self.memory_idx] = muCR
                self.memory_idx = (self.memory_idx + 1) % self.memory_size

            newly_activated_count = 0
            for grid_idx_check in indices_to_process:
                state = self.population[grid_idx_check]
                
                if len(state['improvement_history']) == self.convergence_window:
                    avg_improvement = np.mean(state['improvement_history'])
                    if avg_improvement < self.convergence_threshold:
                        state['status'] = 'converged'
                        if projection_config.get('refining', True):
                            self._refine_single_point(grid_idx_check, use_neighbor_info=True)

                if state['best_fitness'] > (self.global_max_logL - self.roi_threshold):
                    for neighbor_idx in self._get_valid_neighbors(grid_idx_check):
                        if neighbor_idx not in self.population:
                            best_idx = np.argmax(state['fitnesses'])
                            best_warm_start_params = state['continuous_params'][best_idx]
                            best_warm_start_fitness = state['best_fitness']
                            for potential_source_idx in self._get_valid_neighbors(neighbor_idx, include_center=True):
                                if potential_source_idx in self.population:
                                    source_state = self.population[potential_source_idx]
                                    if source_state['best_fitness'] > best_warm_start_fitness:
                                        best_warm_start_fitness = source_state['best_fitness']
                                        source_best_idx = np.argmax(source_state['fitnesses'])
                                        best_warm_start_params = source_state['continuous_params'][source_best_idx]
                            
                            self._activate_point(neighbor_idx, warm_start_params=best_warm_start_params)
                            newly_activated_count += 1

            if self.current_generation % 1 == 0:
                elapsed = time.time() - start_time
                active_count = len([s for s in self.population.values() if s['status'] == 'active'])
                converged_count = len(self.population) - active_count
                
                print(f"Gen {self.current_generation:4d} | Calls: {self.likelihood_calls/1e3:6.1f}k | "
                      f"Grid Pts (act/conv/tot): {active_count:4d}/{converged_count:4d}/{len(self.population):4d} | "
                      f"Global Max logL: {self.global_max_logL:.4e} | "
                      f"New Activations: {newly_activated_count:3d} | "
                      f"Elapsed: {elapsed:.1f}s")
                start_time = time.time()

            if plot_callback and self.current_generation % plot_interval == 0:
                plot_callback(self, fig, axes)
        
        if plot_callback:
            plot_callback(self, fig, axes)

        points_to_skip = []
        patch_iterations = 100
        for i in range(patch_iterations):
            if projection_config.get('patching', True):

                tot_improvement_this_iter, points_attempted_patched, points_successfully_patched = self._run_final_patching(plot_callback=plot_callback, fig=fig, axes=axes, points_to_skip=points_to_skip)

                points_to_reactivate = set()
                for grid_idx in points_to_skip:
                    for dim_idx in range(self.n_proj_dims):
                        for direction in [-1, 1]:
                            offset = np.zeros(self.n_proj_dims, dtype=int)
                            offset[dim_idx] = direction
                            neighbor_idx = tuple(np.array(grid_idx) + offset)
                            if neighbor_idx in points_successfully_patched:
                                points_to_reactivate.add(grid_idx)
                for grid_idx in points_to_reactivate:
                    if grid_idx in points_to_skip:
                        points_to_skip.remove(grid_idx)

                points_to_skip.extend(points_attempted_patched)

                if plot_callback:
                    plot_callback(self, fig, axes)

                if tot_improvement_this_iter < self.patching_conv_threshold:
                    break

        print(f"\n--- Projection on Dims {self.projection_dims} Complete ---")
        print(f"Summary | Calls(tot): {self.likelihood_calls/1e3:.1f}k | Max logL: {self.global_max_logL:.4e}\n")


# --- Test Functions and Plotting ---
# ... (unchanged) ...
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
    # _Anders
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
    import matplotlib.pyplot as plt

    # np.random.seed(15324)
    np.random.seed(498123)

    TEST_FUNCTION = "himmelblau_4d"
    OUTPUT_FILE = "samples.csv"
    
    PROJECTIONS_TO_RUN = [
        {'dims': [0, 1], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [0, 2], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [0, 3], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [1, 2], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [1, 3], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [2, 3], 'grid_points': [100, 100], 'patching': True, 'refining': True},
    ]
    
    log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)
    
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
        refinement_gradient_method="forward-backward-alternate",
        patching_fraction=0.05,
        patching_conv_threshold=0.01,
        memory_size=len(PROJECTIONS_TO_RUN[0]['grid_points']) * 25,
        samples_output_file=OUTPUT_FILE,
    )

    def plot_func_wrapper(s, fig, axes):
        plot_profiles(s, fig, axes)

    sampler.run_projections(
        num_generations=100000,
        max_num_to_evolve=None,
        plot_callback=plot_func_wrapper,
        plot_interval=100,
        skip_init_opt_on_warm_start=False,
    )
