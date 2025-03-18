from mpi4py import MPI
import binminpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def target_function(x):
    # Use an n-dimensional version of the Himmelblau function
    func = 0
    for i in range(len(x)-1):
        func += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    return func


if __name__ == "__main__":

    # Example binning setup for a 3D input space using 5x5x2 bins
    binning_tuples = [(-6.0, 6.0, 5), (-6.0, 6.0, 5), (-6.0, 6.0, 2)]

    #  ==================================================

    # Test "minimize" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'minimize' with MPI parallelization")
        print("----------------------------------------------------")
    result = binminpy.minimize(
        target_function, 
        binning_tuples, 
        method="L-BFGS-B",
        tol=1e-9,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results.
    # Only rank 0 will have the complete result. 
    # For ranks > 0 the 'result' variable is None.
    if rank == 0:
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "differential_evolution" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'differential_evolution' with MPI parallelization")
        print("------------------------------------------------------------------")
    result = binminpy.differential_evolution(
        target_function, 
        binning_tuples, 
        popsize=15,
        tol=0.01,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results
    if rank == 0:    
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "basinhopping" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'basinhopping' with MPI parallelization")
        print("--------------------------------------------------------")
    result = binminpy.basinhopping(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results
    if rank == 0:
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "dual_annealing" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'dual_annealing' with MPI parallelization")
        print("----------------------------------------------------------")
    result = binminpy.dual_annealing(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results
    if rank == 0:    
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "direct" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'direct' with MPI parallelization")
        print("--------------------------------------------------")
    result = binminpy.direct(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results
    if rank == 0:    
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "iminuit" function with MPI parallelization
    if rank == 0:
        print()
        print("# Testing binned 'iminuit' with MPI parallelization")
        print("---------------------------------------------------")
    result = binminpy.iminuit(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="mpi",
        task_distribution="dynamic",
        n_tasks_per_batch=5,
    )

    # Print results
    if rank == 0:    
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()

