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
    # if func < 80:  # "2 sigma"
    #     func = 1
    return func


def guide_function(x, y, *args):
    contour_chi2 = (y - 80.)**2 / 4**2
    return contour_chi2


def bin_check_function(bin_result, x_points, y_points):
    bin_accepted = False
    guide_opt = bin_result.guide_fun
    if guide_opt < 4.0:
        bin_accepted = True
    return bin_accepted



# def target_function(x, *args):
#     """A multi-dimensional version of the Rosenbrock function."""
#     d = len(x)
#     func = 1.0
#     for i in range(0,d-1):
#         func +=  100 * (x[i+1] - x[i] * x[i])**2 + (1 - x[i])**2   
#     # if func < 3.0:  # "2 sigma"
#     #     func = 1.0
#     return func



if __name__ == "__main__":

    # n_bins = 20
    n_bins = 200
    # n_bins = 1
    # dim_combinations = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    dim_combinations = [(0,1)]
    # dim_combinations = [(0,1,2,3)]

    for dim_combo in dim_combinations:

        #
        # Configure and run the optimization
        #

        # binning_tuples = [(-6, 6, 120), (-6, 6, 120), (-6, 6, 5)]
        # binning_tuples = [(-6, 6, 60), (-6, 6, 60), (-6, 6, 60)]
        # binning_tuples = [(-6, 6, 120), (-6, 6, 120), (-6, 6, 30), (-6, 6, 30)]
        # binning_tuples = [(-6, 6, 1), (-6, 6, 1), (-6, 6, 120), (-6, 6, 120)]
        # binning_tuples = [(-6, 6, 10), (-6, 6, 10), (-6, 6, 1), (-6, 6, 1)]

        binning_tuples = [[-6, 6, 1], [-6, 6, 1]]
        # binning_tuples = [[-6, 6, 1], [-6, 6, 1], [-6, 6, 1], [-6, 6, 1]]
        # binning_tuples = [[-5, 10, 1], [-5, 10, 1], [-5, 10, 1], [-5, 10, 1]]
        for dim in dim_combo:
            binning_tuples[dim][2] = n_bins
        binning_tuples = [tuple(l) for l in binning_tuples]

        # binning_tuples = [(-5, 10.0, 200), (-5, 10.0, 200), (-5, 10.0, 200), (-5, 10.0, 200)]
        # binning_tuples = [(-5, 10.0, 100), (-5, 10.0, 100), (-5, 10.0, 100)]
        # binning_tuples = [(-5, 10.0, 60), (-5, 10.0, 60), (-5, 10.0, 60), (-5, 10.0, 60)]
        # binning_tuples = [(-5, 10.0, 30), (-5, 10.0, 30), (-5, 10.0, 120), (-5, 10.0, 120)]


        binned_opt = BinMinBottomUp(
            target_function,
            binning_tuples,
            args=(),
            # guide_function=None,
            guide_function=guide_function,
            # bin_check_function=None,
            bin_check_function=bin_check_function,
            sampler="latinhypercube",
            sampler_kwargs={},
            optimizer="minimize",
            optimizer_kwargs={},
            n_initial_points=100,
            n_sampler_points_per_bin=2,
            # accept_target_below=-np.inf, 
            # accept_delta_target_below=0.0,
            # accept_guide_below=-np.inf, 
            # accept_delta_guide_below=4.0,
            save_evals=True,
            return_evals=False,
            return_bin_centers=True,
            optima_comparison_rtol=1e-6,
            optima_comparison_atol=1e-4,
            n_restarts_per_bin=1,
            n_tasks_per_batch=1,
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



    # Loop over dim combinations done
