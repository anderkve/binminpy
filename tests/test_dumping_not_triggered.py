import numpy as np
from mpi4py import MPI
import binminpy
from binminpy.BinMinBottomUp import BinMinBottomUp
import os
import json

# Basic MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    if rank == 0:
        print("This test needs to be run with 'mpiexec -n N ...' where N >= 2.", flush=True)
    comm.Abort(1)

# Test-specific configurations
TASK_DUMP_FILE = "test_dump_file_case2.json" # A name is provided
MAX_TASKS_IN_MEMORY = float('inf') # Large enough to NOT trigger dumping
N_INITIAL_POINTS = 1
MAX_N_BINS = 5 
ACCEPT_TARGET_BELOW = 0.1

# Simple target function
def simple_target_function(x, *args):
    return (x[0] - 0.75)**2

# Simple guide function
def simple_guide_function(x, y, *args):
    return y

def run_test_dumping_not_triggered():
    if rank == 0:
        print(f"Starting Test Case 2: Dumping Not Triggered (MPI Size: {size})", flush=True)
        # Ensure dump file does not exist before run (it shouldn't be created anyway)
        if os.path.exists(TASK_DUMP_FILE):
            os.remove(TASK_DUMP_FILE)
        assert not os.path.exists(TASK_DUMP_FILE), f"{TASK_DUMP_FILE} should not exist before run."

    comm.Barrier()

    binning_tuples = [[0, 1, 10]]

    binned_opt = BinMinBottomUp(
        simple_target_function,
        binning_tuples,
        guide_function=simple_guide_function,
        n_initial_points=N_INITIAL_POINTS,
        max_tasks_in_memory=MAX_TASKS_IN_MEMORY, # Key: set to not dump
        task_dump_file=TASK_DUMP_FILE,          # Key: file name provided
        accept_target_below=ACCEPT_TARGET_BELOW,
        max_n_bins=MAX_N_BINS,
        neighborhood_distance=1,
        n_sampler_points_per_bin=1,
        print_progress_every_n_batch=1,
        initial_optimizer_kwargs={"tol": 1e-3},
        optimizer_kwargs={"tol": 1e-3}
    )
    
    result = None
    try:
        result = binned_opt.run()
    finally:
        comm.Barrier()
        if rank == 0:
            print("Test Case 2: Assertions (Rank 0)", flush=True)
            
            # Dump file should NOT be created
            assert not os.path.exists(TASK_DUMP_FILE), \
                f"{TASK_DUMP_FILE} should NOT be created if dumping is not triggered."

            assert result is not None, "Optimization should produce a result."
            assert 'y_optimal' in result, "Result should contain 'y_optimal'."
            assert len(result['y_optimal']) > 0, "Should find at least one optimum."
            assert np.isclose(result['y_optimal'][0], 0.0, atol=1e-2), \
                f"Optimal value {result['y_optimal'][0]} not close to 0."
            
            num_bins_processed = len(result.get("bin_tuples", []))
            print(f"Number of bins processed: {num_bins_processed}", flush=True)
            assert num_bins_processed > 0, "Should have processed some bins."

            print("Test Case 2: Dumping Not Triggered - PASSED (Rank 0)", flush=True)
            
            # Cleanup: ensure file is gone even if test failed and it was somehow created
            if os.path.exists(TASK_DUMP_FILE):
                os.remove(TASK_DUMP_FILE)

if __name__ == "__main__":
    run_test_dumping_not_triggered()
    comm.Barrier()
    if rank == 0:
        print("Test Case 2: Completed.", flush=True)
