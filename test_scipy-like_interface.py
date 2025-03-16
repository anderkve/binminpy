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
    binning_tuples = [(-6.0, 6.0, 40), (-6.0, 6.0, 40), (-6.0, 6.0, 2)]

    result = binminpy.minimize(
        target_function, 
        binning_tuples, 
        method="L-BFGS-B",
        tol=1e-9,
        return_evals=False,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-2
    )

    best_bins = result["optimal_bins"]
    print(f"# Global optima found in bin(s) {best_bins}:")
    for i,bin_index_tuple in enumerate(best_bins):
        print(f"- Bin {bin_index_tuple}:")
        print(f"  - x: {result['x_optimal'][i]}")
        print(f"  - y: {result['y_optimal'][i]}")

    print()

