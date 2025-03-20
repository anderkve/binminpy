import numpy as np
from mpi4py import MPI
import binminpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def target_function(x):
    """A multi-dimensional version of the Himmelblau function."""
    func = (   (x[0]**2 + x[1] - 11.)**2 + (x[0] + x[1]**2 - 7.)**2
             + (x[1]**2 + x[2] - 11.)**2 + (x[1] + x[2]**2 - 7.)**2 
             + (x[2]**2 + x[3] - 11.)**2 + (x[2] + x[3]**2 - 7.)**2 )
    return func



if __name__ == "__main__":

    #
    # Configure and run the optimization
    #

    binning_tuples = [(-6.0, 6.0, 120), (-6.0, 6.0, 120), (-6.0, 6.0, 30), (-6.0, 6.0, 30)]

    # Example function for masking some bins
    def bin_masking(bin_centre, bin_edges):
        x0 = bin_centre[0]
        x1 = bin_centre[1]
        ellipse_1 = ((x0 - 0.0) / 4)**2 + ((x1 - 3.0) / 2)**2
        ellipse_2 = ((x0 + 3.5) / 2)**2 + ((x1 - 0.0) / 5)**2
        ellipse_3 = ((x0 - 3.5) / 2)**2 + ((x1 - 0.0) / 4)**2
        if (ellipse_1 > 1) and (ellipse_2 > 1) and (ellipse_3 > 1):
            return False
        return True

    # Do the binned optimization with the scipy.optimize.minimize
    # optimizer and parallelization via multiprocessing.Pool 
    # (parallelization="mpp") using 4 processes.
    result = binminpy.minimize(
        target_function, 
        binning_tuples, 
        return_evals=False,
        return_bin_centers=True,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-4,
        # parallelization="mpi",
        # max_processes=4,
        parallelization="mpi",
        task_distribution="mcmc",
        max_tasks_per_worker=1000 * 20,
        n_tasks_per_batch=20,
        # task_distribution="even",
        # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        method="L-BFGS-B",
        tol=1e-6,
    )

    
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


    #
    # Make plots
    #

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.ticker as ticker
    plt.rcParams.update({'font.size': 14})

    if rank == 0:

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
                              cmap='viridis', norm=LogNorm(vmin=1e0, vmax=2e3),
                              edgecolors='none', shading='flat')
        plt.colorbar(mesh, label='Minimum target function value')
        plt.xlabel(f'$x_{target_dims[0]}$')
        plt.ylabel(f'$x_{target_dims[1]}$')
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[0]))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[1]))
        plt.savefig('plot_2D_x0_x1_MCMC.pdf')


        # Make a 1D plot along dimension 0
        target_dim = 0
        min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dim)
        x_data = result["bin_centers"][min_bin_indices][:,target_dim]
        # x_data = result["x_optimal_per_bin"][min_bin_indices][:,target_dim]  # <-- Use actual best-fit x points, rather than bin centers
        y_data = result["y_optimal_per_bin"][min_bin_indices]
        fig = plt.figure(figsize=(8, 6))
        plt.plot(x_data, y_data, '--', linewidth=1.5, color='0.5')
        plt.plot(x_data, y_data, '.', markersize=10)
        plt.xlim([binning_tuples[target_dim][0], binning_tuples[target_dim][1]])
        plt.ylim([0., 250.])
        plt.xlabel(f'$x_{target_dim}$')
        plt.ylabel(f'Minimum target function value')
        minor_tick_positions = np.linspace(binning_tuples[target_dim][0], binning_tuples[target_dim][1], binning_tuples[target_dim][2] + 1)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_positions))
        plt.savefig('plot_1D_x0_MCMC.pdf')


        # Make a 1D plot along dimension 1
        target_dim = 1
        min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dim)
        x_data = result["bin_centers"][min_bin_indices][:,target_dim]
        # x_data = result["x_optimal_per_bin"][min_bin_indices][:,target_dim]  # <-- Use actual best-fit x points, rather than bin centers
        y_data = result["y_optimal_per_bin"][min_bin_indices]
        plt.figure(figsize=(8, 6))
        plt.plot(x_data, y_data, '--', linewidth=1.5, color='0.5')
        plt.plot(x_data, y_data, '.', markersize=10)
        plt.xlim([binning_tuples[target_dim][0], binning_tuples[target_dim][1]])
        plt.xlabel(f'$x_{target_dim}$')
        plt.ylim([0., 250.])
        plt.ylabel(f'Minimum target function value')
        minor_tick_positions = np.linspace(binning_tuples[target_dim][0], binning_tuples[target_dim][1], binning_tuples[target_dim][2] + 1)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_positions))
        plt.savefig('plot_1D_x1_MCMC.pdf')

