from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_header
import sys

# The number of processes that was used in the binminpy run
mpi_size = int(sys.argv[1])

# Data files
hdf5_file_and_group_names = [ (f"./binminpy_output_rank_{i}.hdf5", "/") for i in range(1,mpi_size) ]

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("y",   ("y", float)),
    ("x0",  ("x0", float)),
    ("x1",  ("x1", float)),
    # ("x2",  ("x2", float)),
    # ("x3",  ("x3", float)),
    # ("x4",  ("x4", float)),
    # ("x5",  ("x5", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets(hdf5_file_and_group_names, datasets, filter_invalid_points=False)

# data = {}
# data["x0"] = result["x_evals"][:,0]
# data["x1"] = result["x_evals"][:,1]
# data["x2"] = result["x_evals"][:,2]
# data["x3"] = result["x_evals"][:,3]

# data["loglike"] = -1 * np.min(result["y_evals"]) - 3.09 / 80. * result["y_evals"]
data["loglike"] = -1 * np.min(data["y"]) - 3.09 / 80. * data["y"]
# data["loglike"] = -1 * np.min(data["y"]) - 3.09 / 20. * data["y"]
n_data_points = len(data["loglike"])

confidence_levels = [0.683, 0.954]
# confidence_levels = []
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

x_keys = [
    "x0", 
    # "x1", 
    # "x2",
    # "x3",
    # "x4",
]
y_keys = [
    "x1", 
    # "x2", 
    # "x3",
    # "x4",
    # "x5",
]
z_keys = [
    "loglike",
]

# dataset_bounds = {}
dataset_bounds = {
    "x0": [-6, 6],
    "x1": [-6, 6],
    "x2": [-6, 6],
    "x3": [-6, 6],
}

# Specify some pretty plot labels?
plot_labels = {
    "x0": "$x_{0}$ (unit)",
    "x1": "$x_{1}$ (unit)",
    "x2": "$x_{2}$ (unit)",
    "x3": "$x_{3}$ (unit)",
    "x4": "$x_{3}$ (unit)",
    "x5": "$x_{3}$ (unit)",
}

# Number of bins used for profiling
# xy_bins = (200, 200)
xy_bins = (160, 160)
# xy_bins = (40, 40)

for z_key in z_keys:
    for x_key in x_keys:
        for y_key in y_keys:

            if x_key == y_key:
                continue 

            # If a pretty plot label is not given, just use the key
            x_label = plot_labels.get(x_key, x_key)
            y_label = plot_labels.get(y_key, y_key)
            z_label = plot_labels.get(z_key, z_key)
            labels = (x_label, y_label, z_label)

            # If variable bounds are not specified, use the full range from the data
            x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
            y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
            xy_bounds = (x_bounds, y_bounds)

            # Copy default GAMBIT plot settings (and make changes if necessary)
            plot_settings = deepcopy(gambit_plot_settings.plot_settings)
            plot_settings["interpolation"] = "none"

            # Create 2D profile likelihood figure
            fig, ax, cbar_ax = plot_utils.plot_2D_profile(
                data[x_key], 
                data[y_key], 
                data[z_key], 
                labels, 
                xy_bins, 
                xy_bounds=xy_bounds, 
                z_is_loglike=True,
                plot_likelihood_ratio=True,
                contour_levels=likelihood_ratio_contour_values,
                z_fill_value = -1*np.finfo(float).max,
                # z_fill_value = np.nan,
                add_max_likelihood_marker = True,
                plot_settings=plot_settings,
            )

            # Add scatter
            # ax.scatter(data[x_key], data[y_key], s=1, color='red', marker='o', linewidths=0.0, alpha=0.01)
            # ax.scatter(data[x_key], data[y_key], s=1, color='red', marker='.', linewidths=0.0, alpha=0.05 * ())

            # Add header
            header_text = f"{n_data_points} points. $1\\sigma$ and $2\\sigma$ CL regions."
            # header_text = f"{n_data_points} points."
            add_header(header_text, ax=ax)

            # Save to file
            output_path = f"./2D_profile__{x_key}__{y_key}__{z_key}.png"
            plot_utils.create_folders_if_not_exist(output_path)
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Wrote file: {output_path}")

            # Make scatter plot
            fig, ax = plot_utils.create_empty_figure_2D(xy_bounds, plot_settings)
            ax.set_facecolor("black")
            ax.scatter(data[x_key], data[y_key], s=1.5, color='yellow', marker='.', linewidths=0.0, alpha=1.0) #alpha=0.05)
            plt.xlabel(plot_labels[x_key], fontsize=plot_settings["fontsize"], labelpad=plot_settings["xlabel_pad"])
            plt.ylabel(plot_labels[y_key], fontsize=plot_settings["fontsize"], labelpad=plot_settings["ylabel_pad"])
            plt.xticks(fontsize=plot_settings["fontsize"])
            plt.yticks(fontsize=plot_settings["fontsize"])
            header_text = f"{n_data_points} points."
            add_header(header_text, ax=ax)
            output_path = f"./2D_scatter__{x_key}__{y_key}.png"
            plot_utils.create_folders_if_not_exist(output_path)
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Wrote file: {output_path}")
