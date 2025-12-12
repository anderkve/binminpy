"""Test that custom MPI communicators work with BinMinBottomUp."""
import numpy as np
from mpi4py import MPI
from binminpy.BinMinBottomUp import BinMinBottomUp

def target_function(x, *args):
    """Simple quadratic function."""
    return np.sum(x**2)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Need at least 2 MPI processes for BinMinBottomUp test")
        exit(0)

    # Test: Pass COMM_WORLD explicitly
    if rank == 0:
        print(f"Test: BinMinBottomUp with explicit COMM_WORLD ({size} processes)")

    binning_tuples = [[-5, 5, 10], [-5, 5, 10]]

    binned_opt = BinMinBottomUp(
        target_function,
        binning_tuples,
        args=(),
        sampler="latinhypercube",
        optimizer="minimize",
        optimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-3},
        sampled_parameters=(0, 1),
        n_initial_points=20,
        n_sampler_points_per_bin=5,
        save_evals=False,
        return_evals=False,
        return_bin_centers=False,
        n_tasks_per_batch=5,
        max_n_bins=20,
        comm=comm,  # Explicit communicator
    )
    result = binned_opt.run()

    if rank == 0:
        print(f"  Global optimum found: x={result['x_optimal'][0]}, y={result['y_optimal'][0]}")
        print(f"  Bins evaluated: {len(result['bin_tuples'])}")
        assert result['y_optimal'][0] < 1.0, "Expected minimum reasonably close to 0"
        print("  PASSED")

    comm.Barrier()

    if rank == 0:
        print("\nAll tests passed!")
