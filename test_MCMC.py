import numpy as np
from mpi4py import MPI
import binminpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.random.seed(15234 + 1235*rank)

def target_function(x, *args):
    """A multi-dimensional version of the Himmelblau function."""
    func = 0
    for i in range(len(x)-1):
        func += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    # if func < 80:  # "2 sigma"
    #     func = 1
    return func

def modified_target_function(x, *args):
    """A multi-dimensional version of the Himmelblau function."""
    func = target_function(x)
    contour_chi2 = (func - 80.)**2 / 4**2
    return contour_chi2


# def target_function(x, *args):
#     """A multi-dimensional version of the Rosenbrock function."""
#     d = len(x)
#     func = 1.0
#     for i in range(0,d-1):
#         func +=  100 * (x[i+1] - x[i] * x[i])**2 + (1 - x[i])**2   
#     # if func < 3.0:  # "2 sigma"
#     #     func = 1.0
#     return func



if __name__ == "__main__":


    binning_tuples = [[-6, 6, 1], [-6, 6, 1], [-6, 6, 1], [-6, 6, 1]]

    result = binminpy.latinhypercube(
        target_function, 
        # modified_target_function, 
        binning_tuples, 
        return_evals=True,
        return_bin_centers=True,
        optima_comparison_rtol=1e-6, 
        optima_comparison_atol=1e-4,
        # parallelization="mpi",
        # max_processes=4,
        parallelization="mpi",
        task_distribution="bottomup",
        mcmc_options={
          # "initial_step_size": 1,
          # "n_tries_before_step_increase": 1*len(binning_tuples),
          # "n_tries_before_jump": 3*len(binning_tuples),
          "always_accept_target_below": -np.inf,  # -np.inf,  
          "always_accept_delta_target_below": 100., # 40.,  # 90.0, #80.,  # 0.
          # "inherit_min_coords": False,
          # "suggestion_cache_size": 1000*size, #5*size,
          "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
        },
        max_tasks_per_worker=750, # int(1000. / len(binning_tuples)) ,
        n_tasks_per_batch=1, # len(binning_tuples),
        n_restarts_per_bin=1,
        # task_distribution="even",
        # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        #
        # latinhypercube options:
        #
        n_hypercube_points=2, # 50
    )


    #
    # Print some results to screen
    #

    if rank == 0:
        best_bins = result["optimal_bins"]
        print(f"# Global optima found in bin(s) {best_bins}:")
        for i,bin_index_tuple in enumerate(best_bins):
            print(f"- Bin {bin_index_tuple}:")
            print(f"  - x: {result['x_optimal'][i]}")
            print(f"  - y: {result['y_optimal'][i]}")
        print()

        n_bins_evaluated = len(result["bin_tuples"])
        max_n_bins = np.prod([bt[2] for bt in binning_tuples])
        print(f"Bins evaluated: {n_bins_evaluated} / {max_n_bins}")
        print()
        print(f"Target function calls: {result['n_target_calls']}")
        print()






    # n_bins = 20
    n_bins = 60
    # n_bins = 1
    # dim_combinations = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    # dim_combinations = [(0,1)]
    dim_combinations = [(0,1,2,3)]

    for dim_combo in dim_combinations:

        #
        # Configure and run the optimization
        #

        # binning_tuples = [(-6, 6, 120), (-6, 6, 120), (-6, 6, 5)]
        # binning_tuples = [(-6, 6, 60), (-6, 6, 60), (-6, 6, 60)]
        # binning_tuples = [(-6, 6, 120), (-6, 6, 120), (-6, 6, 30), (-6, 6, 30)]
        # binning_tuples = [(-6, 6, 1), (-6, 6, 1), (-6, 6, 120), (-6, 6, 120)]
        # binning_tuples = [(-6, 6, 10), (-6, 6, 10), (-6, 6, 1), (-6, 6, 1)]

        # binning_tuples = [[-6, 6, 1], [-6, 6, 1]]
        binning_tuples = [[-6, 6, 1], [-6, 6, 1], [-6, 6, 1], [-6, 6, 1]]
        # binning_tuples = [[-5, 10, 1], [-5, 10, 1], [-5, 10, 1], [-5, 10, 1]]
        for dim in dim_combo:
            binning_tuples[dim][2] = n_bins
        binning_tuples = [tuple(l) for l in binning_tuples]

        # binning_tuples = [(-5, 10.0, 200), (-5, 10.0, 200), (-5, 10.0, 200), (-5, 10.0, 200)]
        # binning_tuples = [(-5, 10.0, 100), (-5, 10.0, 100), (-5, 10.0, 100)]
        # binning_tuples = [(-5, 10.0, 60), (-5, 10.0, 60), (-5, 10.0, 60), (-5, 10.0, 60)]
        # binning_tuples = [(-5, 10.0, 30), (-5, 10.0, 30), (-5, 10.0, 120), (-5, 10.0, 120)]


        # Example function for masking some bins
        def bin_masking(bin_centre, bin_edges):
            x0 = bin_centre[0]
            x1 = bin_centre[1]
            ellipse_1 = ((x0 - 0.0) / 4)**2 + ((x1 - 3.0) / 2)**2
            ellipse_2 = ((x0 + 3.5) / 2)**2 + ((x1 - 0.0) / 5)**2
            ellipse_3 = ((x0 - 3.5) / 2)**2 + ((x1 - 0.0) / 4)**2
            if (ellipse_1 > 1) and (ellipse_2 > 1) and (ellipse_3 > 1):
                return False
            return True


        # result = binminpy.bincenter(
        #     target_function, 
        #     binning_tuples, 
        #     return_evals=True,
        #     return_bin_centers=True,
        #     optima_comparison_rtol=1e-6, 
        #     optima_comparison_atol=1e-4,
        #     # parallelization="mpi",
        #     # max_processes=4,
        #     parallelization="mpi",
        #     task_distribution="bottomup",
        #     mcmc_options={
        #       # "initial_step_size": 1,
        #       # "n_tries_before_step_increase": 1*len(binning_tuples),
        #       # "n_tries_before_jump": 3*len(binning_tuples),
        #       "always_accept_target_below": -np.inf,  # -np.inf,  
        #       "always_accept_delta_target_below": 100.0, #80.,  # 0.
        #       # "inherit_min_coords": False,
        #       # "suggestion_cache_size": 1000*size, #5*size,
        #       "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
        #     },
        #     max_tasks_per_worker=750, # int(1000. / len(binning_tuples)) ,
        #     n_tasks_per_batch=1, # len(binning_tuples),
        #     n_restarts_per_bin=1,
        #     # task_distribution="even",
        #     # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        #     #
        #     # bincenter options:
        #     #
        # )



        result = binminpy.latinhypercube(
            # target_function, 
            modified_target_function, 
            binning_tuples, 
            return_evals=True,
            return_bin_centers=True,
            optima_comparison_rtol=1e-6, 
            optima_comparison_atol=1e-4,
            # parallelization="mpi",
            # max_processes=4,
            parallelization="mpi",
            task_distribution="bottomup",
            mcmc_options={
              # "initial_step_size": 1,
              # "n_tries_before_step_increase": 1*len(binning_tuples),
              # "n_tries_before_jump": 3*len(binning_tuples),
              "always_accept_target_below": -np.inf,  # -np.inf,  
              "always_accept_delta_target_below": 4.0,  #100., # 40.,  # 90.0, #80.,  # 0.
              # "inherit_min_coords": False,
              # "suggestion_cache_size": 1000*size, #5*size,
              "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
            },
            max_tasks_per_worker=750, # int(1000. / len(binning_tuples)) ,
            n_tasks_per_batch=100, # len(binning_tuples),
            n_restarts_per_bin=1,
            # task_distribution="even",
            # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
            #
            # latinhypercube options:
            #
            n_hypercube_points=10, # 50
        )


        # result = binminpy.shgo(
        #     target_function, 
        #     binning_tuples, 
        #     return_evals=False,
        #     return_bin_centers=True,
        #     optima_comparison_rtol=1e-6, 
        #     optima_comparison_atol=1e-4,
        #     # parallelization="mpi",
        #     # max_processes=4,
        #     parallelization="mpi",
        #     task_distribution="bottomup",
        #     mcmc_options={
        #       # "initial_step_size": 1,
        #       # "n_tries_before_step_increase": 1*len(binning_tuples),
        #       # "n_tries_before_jump": 3*len(binning_tuples),
        #       "always_accept_target_below": -np.inf,  # -np.inf,  
        #       "always_accept_delta_target_below": 120.0, #80.,  # 0.
        #       # "inherit_min_coords": False,
        #       # "suggestion_cache_size": 1000*size, #5*size,
        #       "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
        #     },
        #     max_tasks_per_worker=750, # int(1000. / len(binning_tuples)) ,
        #     n_tasks_per_batch=1, # len(binning_tuples),
        #     n_restarts_per_bin=1,
        #     # task_distribution="even",
        #     # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        #     #
        #     # shgo options:
        #     #
        # )


        # result = binminpy.minimize(
        #     target_function, 
        #     binning_tuples, 
        #     return_evals=True,
        #     return_bin_centers=True,
        #     optima_comparison_rtol=1e-6, 
        #     optima_comparison_atol=1e-4,
        #     # parallelization="mpi",
        #     # max_processes=4,
        #     parallelization="mpi",
        #     task_distribution="bottomup",
        #     mcmc_options={
        #       # "initial_step_size": 1,
        #       # "n_tries_before_step_increase": 1*len(binning_tuples),
        #       # "n_tries_before_jump": 3*len(binning_tuples),
        #       "always_accept_target_below": -np.inf,  # -np.inf,  
        #       "always_accept_delta_target_below": 90.0, #80.,  # 0.
        #       # "inherit_min_coords": False,
        #       # "suggestion_cache_size": 1000*size, #5*size,
        #       "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
        #     },
        #     max_tasks_per_worker=750, # int(1000. / len(binning_tuples)) ,
        #     n_tasks_per_batch=1, # len(binning_tuples),
        #     n_restarts_per_bin=1,
        #     # task_distribution="even",
        #     # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        #     # 
        #     # scipy.minimize options:
        #     # 
        #     method="L-BFGS-B",
        #     tol=1e-3,
        #     # options={
        #     #     "maxfun": 2,
        #     #     "maxiter": 2,
        #     # },
        # )



        # result = binminpy.diver(
        #     target_function, 
        #     binning_tuples, 
        #     return_evals=True,
        #     return_bin_centers=True,
        #     optima_comparison_rtol=1e-6, 
        #     optima_comparison_atol=1e-4,
        #     # parallelization="mpi",
        #     # max_processes=4,
        #     parallelization="mpi",
        #     task_distribution="bottomup",
        #     mcmc_options={
        #       # "initial_step_size": 1,
        #       # "n_tries_before_step_increase": 1*len(binning_tuples),
        #       # "n_tries_before_jump": 3*len(binning_tuples),
        #       "always_accept_target_below": -np.inf,  # -np.inf,  
        #       "always_accept_delta_target_below": 100.0, #80.,  # 0.
        #       # "inherit_min_coords": False,
        #       # "suggestion_cache_size": 1000*size, #5*size,
        #       "max_n_bins": int(np.prod([bt[2] for bt in binning_tuples])),
        #     },
        #     max_tasks_per_worker=2500, #250, # int(1000. / len(binning_tuples)) ,
        #     # max_tasks_per_worker=2000, # int(1000. / len(binning_tuples)) ,
        #     n_tasks_per_batch=1, # len(binning_tuples),
        #     n_restarts_per_bin=1,
        #     # task_distribution="even",
        #     # bin_masking=bin_masking,  # <- Activate to use the bin_masking function
        #     #
        #     # diver options:
        #     #
        #     path="diver_output",
        #     nDerived=0,
        #     discrete=np.array([], dtype=np.int32),
        #     partitionDiscrete=False,
        #     maxgen=80, #4,
        #     NP=700*len(binning_tuples), #7000*len(binning_tuples),
        #     F=np.array([0.7]),
        #     Cr=0.9,
        #     lmbda=0.0,
        #     current=False,
        #     expon=False,
        #     bndry=1,
        #     jDE=True,
        #     lambdajDE=False,
        #     convthresh=1e-12,
        #     convsteps=10,
        #     removeDuplicates=True,
        #     savecount=1,
        #     resume=False,
        #     disableIO=True,
        #     outputRaw=False,
        #     outputSam=False,
        #     init_population_strategy=0,
        #     discard_unfit_points=False,
        #     max_initialisation_attempts=10000,
        #     max_acceptable_value=1e6,
        #     seed=-1,
        #     context=None,
        #     verbose=0,
        # )


        
        #
        # Print some results to screen
        #

        if rank == 0:
            best_bins = result["optimal_bins"]
            print(f"# Global optima found in bin(s) {best_bins}:")
            for i,bin_index_tuple in enumerate(best_bins):
                print(f"- Bin {bin_index_tuple}:")
                print(f"  - x: {result['x_optimal'][i]}")
                print(f"  - y: {result['y_optimal'][i]}")
            print()

            n_bins_evaluated = len(result["bin_tuples"])
            max_n_bins = np.prod([bt[2] for bt in binning_tuples])
            print(f"Bins evaluated: {n_bins_evaluated} / {max_n_bins}")
            print()
            print(f"Target function calls: {result['n_target_calls']}")
            print()



    # Loop over dim combinations done

    #
    # Make plots
    #

    if rank == 0:

        from copy import deepcopy
        import numpy as np
        import matplotlib.pyplot as plt

        import gambit_plotting_tools.gambit_plot_utils as plot_utils
        import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
        from gambit_plotting_tools.annotate import add_header

        hdf5_file_and_group_names = [ (f"./binminpy_output_rank_{i}.hdf5", "/") for i in range(1,size) ]

        # Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
        datasets = [
            ("y",   ("y", float)),
            ("x0",  ("x0", float)),
            ("x1",  ("x1", float)),
            ("x2",  ("x2", float)),
            ("x3",  ("x3", float)),
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

        n_data_points = len(data["x0"])
        y_vals = np.zeros(n_data_points)
        for i in range(n_data_points):
            # x = np.array([data["x0"][i], data["x1"][i], data["x2"][i], data["x3"][i]])
            x = np.array([data["x0"][i], data["x1"][i]])
            y_vals[i] = target_function(x)
        data["y"] = y_vals


        # data["loglike"] = -1 * data["y"]
        # data["loglike"] = - 3.09 / 80. * data["y"]
        data["loglike"] = -1 * np.min(data["y"]) - 3.09 / 80. * data["y"]
        # data["loglike"] = -1 * np.min(data["y"]) - 3.09 / 20. * data["y"]

        # confidence_levels = [0.683, 0.954]
        confidence_levels = [0.954]
        # confidence_levels = []
        likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

        x_keys = [
            "x0", 
            "x1", 
            "x2",
            # "x3",
            # "x4",
        ]
        y_keys = [
            "x1", 
            "x2", 
            "x3",
            # "x4",
            # "x5",
        ]
        z_keys = [
            "loglike",
        ]

        dataset_bounds = {}
        # dataset_bounds = {
        #     "x0": [-6, 6],
        #     "x1": [-6, 6],
        #     "x2": [-6, 6],
        #     "x3": [-6, 6],
        # }

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
        xy_bins = (200, 200)
        # xy_bins = (100, 100)
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
                    # ax.scatter(data[x_key], data[y_key], s=1.5, color='yellow', marker='.', linewidths=0.0, alpha=0.05)
                    ax.scatter(data[x_key], data[y_key], s=1.5, color='yellow', marker='.', linewidths=0.0, alpha=0.05)
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




    # -------------------------------------

    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LogNorm
    # import matplotlib.ticker as ticker
    # plt.rcParams.update({'font.size': 14})

    # if rank == 0:

    #     # Make 2D plots

    #     plot_combinations = [(0,1)]
    #     # plot_combinations = [(0,3)]
    #     # plot_combinations = [(1,2)]
    #     # plot_combinations = [(2,3)]
    #     # plot_combinations = [(0,1), (0,2), (1,2)]
    #     # plot_combinations = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    #     for target_dims in plot_combinations:
    #         min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dims)
    #         # x_data = result["bin_centers"][min_bin_indices][:,target_dims]
    #         x_data = result["x_optimal_per_bin"][min_bin_indices][:,target_dims]
    #         y_data = result["y_optimal_per_bin"][min_bin_indices]

    #         bin_limits_per_dim = [np.linspace(binning_tuples[d][0], binning_tuples[d][1], binning_tuples[d][2] + 1) for d in target_dims]
    #         grid_values = np.full((len(bin_limits_per_dim[1]) - 1, len(bin_limits_per_dim[0]) - 1), np.inf)
    #         # grid_values = np.full((len(bin_limits_per_dim[1]) - 1, len(bin_limits_per_dim[0]) - 1), np.max(y_data))
    #         for x, y in zip(x_data, y_data):
    #             x0_idx = np.searchsorted(bin_limits_per_dim[0], x[0], side='right') - 1 
    #             x1_idx = np.searchsorted(bin_limits_per_dim[1], x[1], side='right') - 1 
    #             grid_values[x1_idx, x0_idx] = y

    #         plt.figure(figsize=(8, 6))
    #         mesh = plt.pcolormesh(bin_limits_per_dim[0], bin_limits_per_dim[1], grid_values, 
    #                               # cmap='viridis_r', norm=LogNorm(vmin=1e0, vmax=2e2),
    #                               cmap='viridis_r', vmin=1.0, vmax=120.,
    #                               edgecolors='none', shading='flat')
    #         plt.colorbar(mesh, label='Minimum target function value')
    #         plt.xlabel(f'$x_{target_dims[0]}$')
    #         plt.ylabel(f'$x_{target_dims[1]}$')
    #         ax = plt.gca()
    #         ax.xaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[0]))
    #         ax.yaxis.set_minor_locator(ticker.FixedLocator(bin_limits_per_dim[1]))

    #         # Contour plot?
    #         from scipy.interpolate import griddata
    #         x0_binning_tuple = binning_tuples[target_dims[0]]
    #         x1_binning_tuple = binning_tuples[target_dims[1]]
    #         grid_x0, grid_x1 = np.meshgrid(np.linspace(x0_binning_tuple[0], x0_binning_tuple[1], 2*x0_binning_tuple[2]), np.linspace(x1_binning_tuple[0], x1_binning_tuple[1], 2*x0_binning_tuple[2]))
    #         grid_y = griddata((x_data[:,0], x_data[:,1]), y_data, (grid_x0, grid_x1), method='linear')
    #         ax.contour(grid_x0, grid_x1, grid_y, levels=[80], linewidths=1.5, colors='red')
    #         # ax.contour(grid_x0, grid_x1, grid_y, levels=[3.0], linewidths=1.0, colors='red')

    #         plt.savefig(f"plot_2D_x{target_dims[0]}_x{target_dims[1]}_MCMC.png")


    #     # # Make a 1D plot along dimension 0
    #     # target_dim = 0
    #     # min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dim)
    #     # x_data = result["bin_centers"][min_bin_indices][:,target_dim]
    #     # # x_data = result["x_optimal_per_bin"][min_bin_indices][:,target_dim]  # <-- Use actual best-fit x points, rather than bin centers
    #     # y_data = result["y_optimal_per_bin"][min_bin_indices]
    #     # fig = plt.figure(figsize=(8, 6))
    #     # plt.plot(x_data, y_data, '--', linewidth=1.5, color='0.5')
    #     # plt.plot(x_data, y_data, '.', markersize=10)
    #     # plt.xlim([binning_tuples[target_dim][0], binning_tuples[target_dim][1]])
    #     # plt.ylim([0., 250.])
    #     # plt.xlabel(f'$x_{target_dim}$')
    #     # plt.ylabel(f'Minimum target function value')
    #     # minor_tick_positions = np.linspace(binning_tuples[target_dim][0], binning_tuples[target_dim][1], binning_tuples[target_dim][2] + 1)
    #     # ax = plt.gca()
    #     # ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_positions))
    #     # plt.savefig("plot_1D_x0_MCMC.pdf")


    #     # # Make a 1D plot along dimension 1
    #     # target_dim = 1
    #     # min_bin_indices = binminpy.get_min_bins(result["bin_tuples"], result["y_optimal_per_bin"], target_dims=target_dim)
    #     # x_data = result["bin_centers"][min_bin_indices][:,target_dim]
    #     # # x_data = result["x_optimal_per_bin"][min_bin_indices][:,target_dim]  # <-- Use actual best-fit x points, rather than bin centers
    #     # y_data = result["y_optimal_per_bin"][min_bin_indices]
    #     # plt.figure(figsize=(8, 6))
    #     # plt.plot(x_data, y_data, '--', linewidth=1.5, color='0.5')
    #     # plt.plot(x_data, y_data, '.', markersize=10)
    #     # plt.xlim([binning_tuples[target_dim][0], binning_tuples[target_dim][1]])
    #     # plt.xlabel(f'$x_{target_dim}$')
    #     # plt.ylim([0., 250.])
    #     # plt.ylabel(f'Minimum target function value')
    #     # minor_tick_positions = np.linspace(binning_tuples[target_dim][0], binning_tuples[target_dim][1], binning_tuples[target_dim][2] + 1)
    #     # ax = plt.gca()
    #     # ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_positions))
    #     # plt.savefig("plot_1D_x1_MCMC.pdf")

