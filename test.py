import numpy as np
from binminpy.BinnedOptimizer import BinnedOptimizer
from time import sleep

#
# User target function
#

def Himmelblau(x):
    dim = len(x)
    func = 0
    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    return func

def neg_log_Himmelblau(x):
    func = Himmelblau(x)
    func += 1
    func = -1.0 * np.log(func)
    return func



def target_function(x, a):

    return Himmelblau(x)
    # return neg_log_Himmelblau(x)
    # result = a + np.sum(x**2) 
    # result = a + np.sum(np.sin(x**2)) 

    # sleep(0.5)
    # Waste some time in way that keeps the CPU working
    # for i in range(int(1e6)):
    #     pass

    return result


#
# Test setup
#

n_bins_per_dim = 100
# binning_tuples = [(-2.5, 2.5, 5)]*3
# binning_tuples = [(-6.0, 6.0, n_bins_per_dim)]*2
binning_tuples = [(-6.0, 6.0, n_bins_per_dim), (-6.0, 6.0, n_bins_per_dim), (-6.0, 6.0, 1), (-6.0, 6.0, 1)]
# print(binning_tuples)
# binning_tuples = [(-2.9, -2.7, 1), (3.1, 3.2, 1)]
n_dims = len(binning_tuples)


optimizer_kwargs = {
    "method": "L-BFGS-B",
    "tol": 1e-12,
    "args": (10),
}

max_processes = 10

binned_opt = BinnedOptimizer(
    target_function, 
    binning_tuples,
    optimizer="minimize", 
    # optimizer="differential_evolution", 
    # optimizer="basinhopping", 
    # optimizer="shgo", 
    # optimizer="dual_annealing", 
    # optimizer="direct", 
    optimizer_kwargs=optimizer_kwargs, 
    max_processes=max_processes,
    return_evals=True,
    optima_comparison_rtol=1e-6,
    optima_comparison_atol=1e-2
)

output = binned_opt.run()

print()

n_bins = len(output["all_optimizer_results"])
for i,bin_index_tuple in enumerate(output["bin_order"]):
    print()
    print(f"# Result for bin {bin_index_tuple} (all_optimizer_results[{i}]):")
    print(f"- bin limits: {binned_opt.get_bin_limits(bin_index_tuple)}")
    print(f"- optimization result:\n {output['all_optimizer_results'][i]}")
    # print(f"- x vals:\n {output['all_evals'][i][1]}")
    # print(f"- y vals:\n {output['all_evals'][i][2]}")

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




# 
# Plotting
# 

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
confidence_levels = []
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

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
xy_bins = (n_bins_per_dim, n_bins_per_dim)

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
    # z_is_loglike=True,
    z_is_loglike=False,
    plot_likelihood_ratio=False,
    contour_levels=likelihood_ratio_contour_values,
    contour_coordinates_output_file=f"./plots/2D_profile__{x_key}__{y_key}__{z_key}__coordinates.csv",
    z_fill_value = -1*np.finfo(float).max,
    add_max_likelihood_marker = True,
    plot_settings=plot_settings,
)

# Add text
# fig.text(0.525, 0.350, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="white")

# # Add anything else to the plot, e.g. some more lines and labels and stuff
# ax.plot([20.0, 30.0], [5.0, 3.0], color="white", linewidth=plot_settings["contour_linewidth"], linestyle="dashed")
# fig.text(0.53, 0.79, "A very important line!", ha="left", va="center", rotation=-31.5, fontsize=plot_settings["fontsize"]-5, color="white")

# # Draw a contour using coordinates stored in a .csv file
# x_contour, y_contour = np.loadtxt("./example_data/contour_coordinates.csv", delimiter=",", usecols=(0, 1), unpack=True)
# ax.plot(x_contour, y_contour, color="orange", linestyle="dashed", linewidth=plot_settings["contour_linewidth"], alpha=0.7)
# fig.text(0.23, 0.23, "Overlaid contour from\n coordinates in some data file", ha="left", va="center", fontsize=plot_settings["fontsize"]-5, color="orange")

# Save to file
output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")




