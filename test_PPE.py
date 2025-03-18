import numpy as np
import binminpy

def target_function(x):
    # Use an three-dimensional version of the Himmelblau function
    func = (  (x[0]**2 + x[1] - 11.)**2 + (x[0] + x[1]**2 - 7.)**2
            + (x[1]**2 + x[2] - 11.)**2 + (x[1] + x[2]**2 - 7.)**2 )
    return func


if __name__ == "__main__":

    # Example binning setup for a 3D input space using 5x5x2 bins
    binning_tuples = [(-6.0, 6.0, 5), (-6.0, 6.0, 5), (-6.0, 6.0, 2)]

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


    #  ==================================================


    # Test "iminuit" function with PPE parallelization
    print()
    print("# Testing binned 'iminuit' with PPE parallelization")
    print("---------------------------------------------------")
    result = binminpy.iminuit(
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


    # Test "diver" function with PPE parallelization
    print()
    print("# Testing binned 'diver' with PPE parallelization")
    print("-------------------------------------------------")
    result = binminpy.diver(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
        # diver options:
        path="diver_output",
        nDerived=0,
        discrete=np.array([], dtype=np.int32),
        partitionDiscrete=False,
        maxgen=300,
        NP=max(10*len(binning_tuples), 5),
        F=np.array([0.7]),
        Cr=0.9,
        lmbda=0.0,
        current=False,
        expon=False,
        bndry=1,
        jDE=True,
        lambdajDE=True,
        convthresh=1e-3,
        convsteps=10,
        removeDuplicates=True,
        savecount=1,
        resume=False,
        disableIO=True,
        outputRaw=False,
        outputSam=False,
        init_population_strategy=0,
        discard_unfit_points=False,
        max_initialisation_attempts=10000,
        max_acceptable_value=1e6,
        seed=-1,
        context=None,
        verbose=0,
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


    # Test "adam" function with PPE parallelization
    print()
    print("# Testing binned 'adam' with PPE parallelization")
    print("------------------------------------------------")
    result = binminpy.adam(
        target_function, 
        binning_tuples, 
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2,
        parallelization="ppe",
        max_processes=4,
        # Adam options:
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
    )

    # Print results
    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")
    print()
