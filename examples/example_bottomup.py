import numpy as np
from mpi4py import MPI
import binminpy
from binminpy.BinMinBottomUp import BinMinBottomUp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def target_function(x, *args):
    # """A 2D Himmelblau function in (x0,x1) with x2 representing a shift of x1"""
    func = (x[0]**2 + x[1] + x[2] - 11.)**2 + (x[0] + (x[1] + x[2])**2 - 7.)**2
    return func


def bin_check_function(bin_result, x_points, y_points):
    bin_accepted = bool(bin_result.fun < 30.0)
    return bin_accepted


def callback(bin_result, x_points, y_points):
    print(f"callback: rank {rank}: This is the callback function. Got {len(x_points)} points.", flush=True)


def set_eval_points(bin_index_tuple, bounds):
    print(f"set_eval_points: rank {rank}: This is the set_eval_points function. Got bin {bin_index_tuple} with bounds {bounds}.", flush=True)
    x_lower_lims = np.array([b[0] for b in bounds])
    x_upper_lims = np.array([b[1] for b in bounds])
    n_pts = 2
    n_dims = len(bounds)
    points = x_lower_lims + np.random.random((n_pts, n_dims)) * (x_upper_lims - x_lower_lims)
    return points



if __name__ == "__main__":

    binning_tuples = [[-6, 6, 120], [-6, 6, 120], [-0.5, 0.5, 2]]

    binned_opt = BinMinBottomUp(
        target_function,
        binning_tuples,
        args=(),
        guide_function=None,
        bin_check_function=bin_check_function,
        # callback=callback,
        # callback_on_rank_0=True,
        sampler="latinhypercube",
        optimizer="minimize",
        optimizer_kwargs={
            "tol": 1e-9,
            "method": "L-BFGS-B",
        },
        sampled_parameters=(),
        # set_eval_points=set_eval_points,
        # set_eval_points_on_rank_0=True,
        # initial_optimizer="minimize",
        # n_initial_points=10,
        initial_optimizer="differential_evolution",
        n_initial_points=1,
        n_sampler_points_per_bin=1,
        inherit_best_init_point_within_bin=False,
        # accept_target_below=-np.inf, 
        # accept_delta_target_below=0.0,
        # accept_guide_below=-np.inf, 
        # accept_delta_guide_below=4.0,
        save_evals=False,
        return_evals=False,
        return_bin_centers=True,
        optima_comparison_rtol=1e-6,
        optima_comparison_atol=1e-4,
        neighborhood_distance=1,
        n_optim_restarts_per_bin=1,
        n_tasks_per_batch=10,
        print_progress_every_n_batch=1,
        max_tasks_per_worker=np.inf,
        max_n_bins=np.inf,
    )
    result = binned_opt.run()


    #
    # Print some results to screen
    #

    if rank == 0:
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()

        n_bins_evaluated = len(result["bin_tuples"])
        max_n_bins = np.prod([bt[2] for bt in binning_tuples])
        print(f"Bins evaluated: {n_bins_evaluated} / {max_n_bins}")
        print()
        print(f"Target function calls: {result['n_target_calls']}")
        print()


    #
    # Make plots
    #

    if rank == 0:

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import matplotlib.ticker as ticker
        plt.rcParams.update({'font.size': 14})

        # Make a 2D plot in dimensions 0 and 1
        target_dims = [0,1]
        min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dims)
        x_data_centered = result["bin_centers"][min_bin_indices][:,target_dims]
        y_data = result["y_optimal_per_bin"][min_bin_indices]

        bin_limits_per_dim = [np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1) for d in target_dims]
        grid_values = np.full((len(bin_limits_per_dim[1]) - 1, len(bin_limits_per_dim[0]) - 1), np.inf)
        for x, y in zip(x_data_centered, y_data):
            x0_idx = np.searchsorted(bin_limits_per_dim[0], x[0], side='right') - 1 
            x1_idx = np.searchsorted(bin_limits_per_dim[1], x[1], side='right') - 1 
            grid_values[x1_idx, x0_idx] = y

        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(bin_limits_per_dim[0], bin_limits_per_dim[1], grid_values, 
                              cmap='viridis', vmin=0, vmax=30,
                              edgecolors='none', shading='flat')
        plt.colorbar(mesh, label='Optimized target value in bin')
        plt.xlabel(f'$x_{target_dims[0]}$')
        plt.ylabel(f'$x_{target_dims[1]}$')
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[0][::2]))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[1][::2]))
        plt.savefig('plot_2D_x0_x1.pdf')
