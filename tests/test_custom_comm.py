"""Test that custom MPI communicators work correctly."""
import numpy as np
from mpi4py import MPI
from binminpy.BinMinMPI import BinMinMPI

def target_function(x):
    """Simple quadratic function."""
    return np.sum(x**2)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test 1: Pass COMM_WORLD explicitly (should work same as default)
    if rank == 0:
        print(f"Test 1: Using explicit COMM_WORLD with {size} processes")

    binning_tuples = [(-5.0, 5.0, 3), (-5.0, 5.0, 3)]

    binned_opt = BinMinMPI(
        target_function,
        binning_tuples,
        optimizer="minimize",
        optimizer_kwargs={"method": "L-BFGS-B"},
        return_bin_results=True,
        comm=comm,  # Explicit communicator
    )
    result = binned_opt.run()

    if rank == 0:
        print(f"  Global optimum found: x={result['x_optimal'][0]}, y={result['y_optimal'][0]}")
        print(f"  Bins evaluated: {len(result['bin_tuples'])}")
        assert abs(result['y_optimal'][0]) < 1e-6, "Expected minimum near 0"
        print("  PASSED")

    comm.Barrier()

    # Test 2: Create a subcommunicator and use it
    if size >= 2:
        # Split into two groups: even and odd ranks
        color = rank % 2
        subcomm = comm.Split(color=color, key=rank)
        subrank = subcomm.Get_rank()
        subsize = subcomm.Get_size()

        if rank == 0:
            print(f"\nTest 2: Using subcommunicator (even ranks only, {subsize} processes)")

        # Only even ranks run this
        if color == 0:
            binned_opt2 = BinMinMPI(
                target_function,
                binning_tuples,
                optimizer="minimize",
                optimizer_kwargs={"method": "L-BFGS-B"},
                return_bin_results=True,
                comm=subcomm,  # Use subcommunicator
            )
            result2 = binned_opt2.run()

            if subrank == 0:
                print(f"  Global optimum found: x={result2['x_optimal'][0]}, y={result2['y_optimal'][0]}")
                print(f"  Bins evaluated: {len(result2['bin_tuples'])}")
                assert abs(result2['y_optimal'][0]) < 1e-6, "Expected minimum near 0"
                print("  PASSED")

        subcomm.Free()

    comm.Barrier()

    if rank == 0:
        print("\nAll tests passed!")
