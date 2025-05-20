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
TASK_DUMP_FILE = "test_dump_file_case3.json"
# max_tasks_in_memory can be anything, as we are testing loading, not dumping.
# Let's set it so it wouldn't dump if new tasks were generated from these.
MAX_TASKS_IN_MEMORY = 10 
N_INITIAL_POINTS = 0 # No initial points, rely on dump file
MAX_N_BINS = 5       # Process at least the tasks from the file + some neighbors
ACCEPT_TARGET_BELOW = 0.5 # Acceptable value to consider bins for neighborhood

# Tasks to pre-populate the dump file with
# For binning_tuples = [[0, 1, 10]], tasks are like (idx,)
PREEXISTING_TASKS = [(1,), (2,), (3,)] 

# Simple target function
def simple_target_function(x, *args):
    return (x[0] - 0.25)**2 # Minimum at x=0.25 (bin 2 if 10 bins from 0-1)

# Simple guide function
def simple_guide_function(x, y, *args):
    return y

def run_test_loading_from_preexisting():
    if rank == 0:
        print(f"Starting Test Case 3: Loading from Pre-existing (MPI Size: {size})", flush=True)
        
        # Manually create and populate the dump file
        print(f"Creating pre-existing dump file: {TASK_DUMP_FILE}", flush=True)
        with open(TASK_DUMP_FILE, 'w') as f:
            for task_tuple in PREEXISTING_TASKS:
                f.write(json.dumps(task_tuple) + '\n')
        assert os.path.exists(TASK_DUMP_FILE), f"{TASK_DUMP_FILE} should exist before run."

    comm.Barrier() # Ensure file is created before any rank proceeds

    binning_tuples = [[0, 1, 10]]

    binned_opt = BinMinBottomUp(
        simple_target_function,
        binning_tuples,
        guide_function=simple_guide_function,
        n_initial_points=N_INITIAL_POINTS,      # Key: 0 initial points
        max_tasks_in_memory=MAX_TASKS_IN_MEMORY,
        task_dump_file=TASK_DUMP_FILE,          # Key: file name provided
        accept_target_below=ACCEPT_TARGET_BELOW, 
        max_n_bins=MAX_N_BINS, # Should be >= len(PREEXISTING_TASKS) + neighbors
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
            print("Test Case 3: Assertions (Rank 0)", flush=True)
            
            # Dump file should be deleted after tasks are loaded
            assert not os.path.exists(TASK_DUMP_FILE), \
                f"{TASK_DUMP_FILE} should be deleted after tasks are loaded."

            assert result is not None, "Optimization should produce a result."
            assert 'y_optimal' in result, "Result should contain 'y_optimal'."
            
            # If PREEXISTING_TASKS were (1,), (2,), (3,) for (x-0.25)^2, 
            # bin 2 (0.2 to 0.3) contains the true minimum.
            # So, the optimization should find a value close to 0.
            assert len(result['y_optimal']) > 0, "Should find at least one optimum."
            assert np.isclose(result['y_optimal'][0], 0.0, atol=1e-2), \
                f"Optimal value {result['y_optimal'][0]} not close to 0. Target min at x=0.25"

            num_bins_processed = len(result.get("bin_tuples", []))
            print(f"Number of bins processed: {num_bins_processed}", flush=True)
            # It should process at least the initial tasks, and potentially their neighbors if they are "nice"
            # up to max_n_bins.
            assert num_bins_processed >= len(PREEXISTING_TASKS) or num_bins_processed == MAX_N_BINS, \
                f"Should have processed at least {len(PREEXISTING_TASKS)} bins or MAX_N_BINS ({MAX_N_BINS}). Got: {num_bins_processed}"


            print("Test Case 3: Loading from Pre-existing - PASSED (Rank 0)", flush=True)
            
            # Cleanup again just in case
            if os.path.exists(TASK_DUMP_FILE):
                os.remove(TASK_DUMP_FILE)

if __name__ == "__main__":
    run_test_loading_from_preexisting()
    comm.Barrier()
    if rank == 0:
        print("Test Case 3: Completed.", flush=True)
