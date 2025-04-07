import numpy as np
from mpi4py import MPI
import binminpy
from binminpy.BinMinBottomUp import BinMinBottomUp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def target_function(x, *args):
    """A 2D Himmelblau function in (x0,x1)"""
    func = 0
    func += (x[0]**2 + x[1] - 11.)**2 + (x[0] + x[1]**2 - 7.)**2
    return func


def guide_function(x, y, *args):
    contour_chi2 = (y - 80.)**2 / 10**2
    return contour_chi2


def bin_check_function(bin_result, x_points, y_points):
    bin_accepted = False
    guide_opt = bin_result.guide_fun
    if guide_opt < 1.0:
        bin_accepted = True
    return bin_accepted


if __name__ == "__main__":

    binning_tuples = [[-6, 6, 30], [-6, 6, 30]]

    binned_opt = BinMinBottomUp(
        target_function,
        binning_tuples,
        args=(),
        guide_function=guide_function,
        bin_check_function=bin_check_function,
        # callback=callback,
        # callback_on_rank_0=True,
        sampler="latinhypercube",
        optimizer="minimize",
        optimizer_kwargs={
            "tol": 1e-9,
            "method": "L-BFGS-B",
        },
        sampled_parameters=(0,1),
        optimized_parameters=(),
        # set_eval_points=set_eval_points,
        # set_eval_points_on_rank_0=True,
        n_initial_points=100,
        n_sampler_points_per_bin=10,
        inherit_best_init_point_within_bin=False,
        # accept_target_below=-np.inf, 
        # accept_delta_target_below=0.0,
        # accept_guide_below=-np.inf, 
        # accept_delta_guide_below=4.0,
        save_evals=False,
        return_evals=True,
        return_bin_centers=False,
        optima_comparison_rtol=1e-6,
        optima_comparison_atol=1e-4,
        neighborhood_distance=1,
        n_optim_restarts_per_bin=1,
        n_tasks_per_batch=10,
        print_progress_every_n_batch=100,
        max_tasks_per_worker=np.inf,
        max_n_bins=np.inf,
    )
    result = binned_opt.run()


    #
    # Make plots
    #

    if rank == 0:

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import matplotlib.ticker as ticker
        plt.rcParams.update({'font.size': 14})

        # Make a binned 2D plot in dimensions 0 and 1
        target_dims = [0,1]
        min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dims)
        x_data = result["x_evals"][:,target_dims]
        y_data = result["y_evals"]

        bin_limits_per_dim = [np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1) for d in target_dims]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_data[:,0], x_data[:,1], s=12, c=y_data, vmin=65, vmax=95,
                    linewidth=0.05, edgecolors="0.0", cmap="RdYlBu")
        plt.colorbar(label='Sample target value')
        plt.xlabel(f'$x_{target_dims[0]}$')
        plt.ylabel(f'$x_{target_dims[1]}$')
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[0]))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[1]))
        plt.savefig('plot_2D_x0_x1.pdf')

