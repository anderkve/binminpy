import binminpy

def target_function(x):
    # Use an n-dimensional version of the Himmelblau function
    dim = len(x)
    func = 0
    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    return func


if __name__ == "__main__":

    # Example binning setup for a 3D input space
    # binning_tuples = [(-6.0, 6.0, 40), (-6.0, 6.0, 40), (-6.0, 6.0, 2)]
    binning_tuples = [(-6.0, 6.0, 4), (-6.0, 6.0, 4), (-6.0, 6.0, 2)]

    #  ==================================================

    # Test "minimize" function without parallelization
    print()
    print("# Testing binned 'minimize' without parallelization")
    print("---------------------------------------------------")
    result = binminpy.minimize(
        target_function, 
        binning_tuples, 
        method="L-BFGS-B",
        tol=1e-9,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization=None,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()


    #  ==================================================


    # Test "minimize" function with PPE parallelization
    print()
    print("# Testing binned 'minimize' with PPE parallelization")
    print("----------------------------------------------------")
    result = binminpy.minimize(
        target_function, 
        binning_tuples, 
        method="L-BFGS-B",
        tol=1e-9,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
    )

    # Print results.
    # Only rank 0 will have the complete result. 
    # For ranks > 0 the 'result' variable is None.
    if result is not None:
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()


    #  ==================================================


    # Test "differential_evolution" function with PPE parallelization
    print()
    print("# Testing binned 'differential_evolution' with PPE parallelization")
    print("------------------------------------------------------------------")
    result = binminpy.differential_evolution(
        target_function, 
        binning_tuples, 
        popsize=15,
        tol=0.01,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()


    #  ==================================================


    # Test "basinhopping" function with PPE parallelization
    print()
    print("# Testing binned 'basinhopping' with PPE parallelization")
    print("--------------------------------------------------------")
    result = binminpy.basinhopping(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()


    #  ==================================================


    # Test "dual_annealing" function with PPE parallelization
    print()
    print("# Testing binned 'dual_annealing' with PPE parallelization")
    print("----------------------------------------------------------")
    result = binminpy.dual_annealing(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()


    #  ==================================================


    # Test "direct" function with PPE parallelization
    print()
    print("# Testing binned 'direct' with PPE parallelization")
    print("--------------------------------------------------")
    result = binminpy.direct(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()


