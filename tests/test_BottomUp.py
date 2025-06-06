import numpy as np
from mpi4py import MPI
import binminpy
from binminpy.BinMinBottomUp import BinMinBottomUp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.random.seed(15234 + 1235*rank)

def target_function(x, *args):
    """A multi-dimensional version of the Himmelblau function."""
    func = 0
    for i in range(len(x)-1):
        func += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    return func


def guide_function(x, y, *args):
    contour_chi2 = (y - 80.)**2 / 10**2
    return contour_chi2


def bin_check_function(bin_result, x_points, y_points):
    bin_accepted = False
    guide_fun = bin_result.guide_fun
    if guide_fun < 1.0:
        bin_accepted = True
    # target_opt = bin_result.fun
    # if target_opt < 100.0:
    #     bin_accepted = True
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


    # binning_tuples = [[-6, 6, 50], [-6, 6, 50], [-6, 6, 50], [-6, 6, 50]]
    binning_tuples = [[-6, 6, 100], [-6, 6, 100]]

    binned_opt = BinMinBottomUp(
        target_function,
        binning_tuples,
        args=(),
        # guide_function=None,
        guide_function=guide_function,
        # bin_check_function=None,
        bin_check_function=bin_check_function,
        # callback=callback,
        # callback_on_rank_0=True,
        sampler="latinhypercube",
        optimizer="minimize",
        optimizer_kwargs={
            "tol": 1e-3,
            "method": "L-BFGS-B",
        },
        # sampled_parameters=(0,1,2,3),
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
        save_evals=True,
        return_evals=False,
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
