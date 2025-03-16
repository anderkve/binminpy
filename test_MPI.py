import numpy as np
from binminpy.BinnedOptimizerMPI import BinnedOptimizerMPI


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
    n_dims = len(binning_tuples)

    # Specify the optimizer and any kwargs.
    optimizer = "minimize" # "minimize", "differential_evolution", "basinhopping", "shgo", "dual_annealing", "direct"
    optimizer_kwargs = {
        "method": "L-BFGS-B",
        "tol": 1e-9,
    }

    # Instantiate the BinnedOptimizerMPI.
    binned_opt = BinnedOptimizerMPI(
        target_function,
        binning_tuples,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        return_evals=True,
        optima_comparison_rtol=1e-6,
        optima_comparison_atol=1e-2
    )

    # Run the optimization.
    output = binned_opt.run()

    # Only rank 0 will have the complete result. 
    # For ranks > 0 the 'output' variable is None.
    if output is not None:

        # Print output
        print()
        n_bins = len(output["all_optimizer_results"])
        for i,bin_index_tuple in enumerate(output["bin_order"]):
            print()
            print(f"# Result for bin {bin_index_tuple} (all_optimizer_results[{i}]):")
            print(f"- bin limits: {binned_opt.get_bin_limits(bin_index_tuple)}")
            print(f"- optimization result:\n {output['all_optimizer_results'][i]}")

        print()
        print()

        best_bins = output["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - bin limits: {binned_opt.get_bin_limits(bin_index_tuple)}")
            print(f"  - x: {output['x_optimal'][i]}")
            print(f"  - y: {output['y_optimal'][i]}")

        print()


        # Make plots
        from copy import deepcopy
        import matplotlib.pyplot as plt
        import gambit_plotting_tools.gambit_plot_utils as plot_utils
        import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings

        x_points = output["x_evals"]
        y_points = output["y_evals"]

        n_dims = x_points.shape[1]
        data = {}
        for i in range(n_dims):
            data[f"x{i}"] = x_points[:,i]
        data["y"] = y_points
        data["neg_loglike"] = np.log(y_points + np.abs(np.min(y_points)) + 1.0)
        data["loglike"] = -1.0 * data["neg_loglike"]

        # confidence_levels = [0.683, 0.954]
        # likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)
        max_loglike = np.max(data["loglike"])
        contour_levels = [max_loglike - i for i in range(5)]

        # Plot variables
        x_key = "x0"
        y_key = "x1"
        z_key = "loglike"

        # Set some bounds manually?
        dataset_bounds = {
            "x0": [-6, 6],
            "x1": [-6, 6],
            "x2": [-6, 6],
        }

        # Specify some pretty plot labels?
        plot_labels = {
            "x0": "$x_{0}$ (unit)",
            "x1": "$x_{1}$ (unit)",
            "x2": "$x_{2}$ (unit)",
        }

        # Number of bins used for profiling
        xy_bins = (binning_tuples[0][2], binning_tuples[1][2])

        # Load default plot settings (and make adjustments if necessary)
        plot_settings = deepcopy(gambit_plot_settings.plot_settings)
        plot_settings["interpolation"] = "none"

        # If variable bounds are not specified in dataset_bounds, use the full range from the data
        x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
        y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
        xy_bounds = (x_bounds, y_bounds)

        # If a pretty plot label is not given, just use the key
        x_label = plot_labels.get(x_key, x_key)
        y_label = plot_labels.get(y_key, y_key)
        z_label = plot_labels.get(z_key, z_key)
        labels = (x_label, y_label, z_label)

        # Create 2D profile likelihood figure
        fig, ax, cbar_ax = plot_utils.plot_2D_profile(
            data[x_key], 
            data[y_key], 
            data[z_key], 
            labels, 
            xy_bins, 
            xy_bounds=xy_bounds, 
            z_is_loglike=False,
            plot_likelihood_ratio=False,
            # contour_levels=likelihood_ratio_contour_values,
            contour_levels=contour_levels,
            contour_coordinates_output_file=f"./plots/2D_profile__{x_key}__{y_key}__{z_key}__coordinates.csv",
            z_fill_value = -1*np.finfo(float).max,
            add_max_likelihood_marker = True,
            plot_settings=plot_settings,
        )

        # Save to file
        output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
        plot_utils.create_folders_if_not_exist(output_path)
        plt.savefig(output_path)
        plt.close()
        print(f"Wrote file: {output_path}")




