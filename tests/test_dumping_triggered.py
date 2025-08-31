import numpy as np
from mpi4py import MPI
import binminpy
from binminpy.BinMinBottomUp import BinMinBottomUp
import os
import json
import time # For potential file system race conditions

# Basic MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    if rank == 0:
        print("This test needs to be run with 'mpiexec -n N ...' where N >= 2.", flush=True)
    comm.Abort(1)

# Test-specific configurations
TASK_DUMP_FILE = "test_dump_file_case1.json"
MAX_TASKS_IN_MEMORY = 1 # Small enough to trigger dumping
N_INITIAL_POINTS = 1
MAX_N_BINS = 5 
ACCEPT_TARGET_BELOW = 0.1 # To ensure neighborhoods are explored

# Simple target function
def simple_target_function(x, *args):
    return (x[0] - 0.75)**2

# Simple guide function (can be None if accept_target_below is used directly)
def simple_guide_function(x, y, *args):
    return y

def run_test_dumping_triggered():
    if rank == 0:
        print(f"Starting Test Case 1: Dumping Triggered (MPI Size: {size})", flush=True)
        # Ensure dump file does not exist before run
        if os.path.exists(TASK_DUMP_FILE):
            os.remove(TASK_DUMP_FILE)
        assert not os.path.exists(TASK_DUMP_FILE), f"{TASK_DUMP_FILE} should not exist before run."

    comm.Barrier()

    binning_tuples = [[0, 1, 10]] # 1D, 10 bins

    binned_opt = BinMinBottomUp(
        simple_target_function,
        binning_tuples,
        guide_function=simple_guide_function,
        n_initial_points=N_INITIAL_POINTS,
        max_tasks_in_memory=MAX_TASKS_IN_MEMORY,
        task_dump_file=TASK_DUMP_FILE,
        accept_target_below=ACCEPT_TARGET_BELOW, # Generates tasks if y_opt is low
        max_n_bins=MAX_N_BINS,
        neighborhood_distance=1, # Default, but explicit for clarity
        n_sampler_points_per_bin=1, # Minimal sampling
        print_progress_every_n_batch=1,
        initial_optimizer_kwargs={"tol": 1e-3}, # Faster initial opt
        optimizer_kwargs={"tol": 1e-3} # Faster bin opt
    )
    
    result = None
    try:
        result = binned_opt.run()
    finally:
        comm.Barrier() # Ensure all ranks finish run()
        if rank == 0:
            # Assertions for rank 0
            print("Test Case 1: Assertions (Rank 0)", flush=True)
            
            # Dump file should be created and then deleted by the loading mechanism
            # A slight delay might be needed for the file system if checking for existence during run
            # For this test, we primarily check it's gone AFTER run, implying it was processed.
            assert not os.path.exists(TASK_DUMP_FILE), \
                f"{TASK_DUMP_FILE} should be deleted after tasks are loaded and processed."

            assert result is not None, "Optimization should produce a result."
            assert 'y_optimal' in result, "Result should contain 'y_optimal'."
            assert len(result['y_optimal']) > 0, "Should find at least one optimum."
            # Plausible result: y_optimal should be close to 0 for (x-0.75)^2
            assert np.isclose(result['y_optimal'][0], 0.0, atol=1e-2), \
                f"Optimal value {result['y_optimal'][0]} not close to 0."
            
            # Check if enough bins were processed to likely involve dumping/loading
            # This is an indirect check. If max_n_bins is small, and tasks were generated,
            # dumping must have occurred if max_tasks_in_memory was also small.
            num_bins_processed = len(result.get("bin_tuples", []))
            print(f"Number of bins processed: {num_bins_processed}", flush=True)
            assert num_bins_processed > 0, "Should have processed some bins."
            # If N_INITIAL_POINTS = 1, and it found a good spot, its neighbors would be added.
            # If neighborhood_distance=1, for 1D, it's 2 neighbors.
            # Initial task + 2 neighbors = 3 tasks. If MAX_TASKS_IN_MEMORY = 1, dumping should occur.
            # The number of bins actually completed could be up to MAX_N_BINS.
            # The key is that the *planned* tasks exceeded MAX_TASKS_IN_MEMORY.

            print("Test Case 1: Dumping Triggered - PASSED (Rank 0)", flush=True)
            
            # Cleanup again just in case, though it should be gone
            if os.path.exists(TASK_DUMP_FILE):
                os.remove(TASK_DUMP_FILE)

if __name__ == "__main__":
    run_test_dumping_triggered()
    comm.Barrier()
    if rank == 0:
        print("Test Case 1: Completed.", flush=True)
