"""
binminpy
========

A package for parallelised, binned function optimisation.
"""

__version__ = "0.1.0"
__author__ = "Anders Kvellestad"
__credits__ = ""


import numpy as np


# Helper functions

def _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                   optima_comparison_rtol, optima_comparison_atol, parallelization,
                   max_processes, task_distribution, n_tasks_per_batch, bin_masking):
    """Helper function to start the optimizer with the requested parallelization. """

    # Check the parallelization argument.
    if parallelization is None:
        parallelization = "serial"
    if parallelization.lower() not in ["serial", "ppe", "mpi"]:
        raise Exception(f"Unknown setting for argument 'parallelization' ('{task_distribution}'). Valid options are 'serial'/None, 'ppe' and 'mpi'.")
    parallelization = parallelization.lower()        

    # Run binned optimization with the requested parallelization.
    if parallelization == "serial":
        from binminpy.BinnedOptimizer import BinnedOptimizer
        
        binned_opt = BinnedOptimizer(
            fun,
            binning_tuples,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            return_evals=return_evals,
            optima_comparison_rtol=optima_comparison_rtol,
            optima_comparison_atol=optima_comparison_atol,
            bin_masking=bin_masking,
        )
        output = binned_opt.run()
        return output

    elif parallelization == "ppe":
        from binminpy.BinnedOptimizerPPE import BinnedOptimizerPPE
        binned_opt = BinnedOptimizerPPE(
            fun,
            binning_tuples,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            return_evals=return_evals,
            optima_comparison_rtol=optima_comparison_rtol,
            optima_comparison_atol=optima_comparison_atol,
            max_processes=max_processes,
            bin_masking=bin_masking,            
        )
        output = binned_opt.run()
        return output

    elif parallelization == "mpi":
        from binminpy.BinnedOptimizerMPI import BinnedOptimizerMPI
        binned_opt = BinnedOptimizerMPI(
            fun,
            binning_tuples,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            return_evals=return_evals,
            optima_comparison_rtol=optima_comparison_rtol,
            optima_comparison_atol=optima_comparison_atol,
            task_distribution=task_distribution,
            n_tasks_per_batch=n_tasks_per_batch,
            bin_masking=bin_masking,
        )
        output = binned_opt.run()
        return output

    else:
        raise Exception("This should never happen. Please report this as a bug.")



# Below is a collection of functions to allow using binminpy 
# through an interface similar to scipy.optimize.

def minimize(fun, binning_tuples, return_evals=False, 
             optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
             parallelization=None, max_processes=1, task_distribution="even", 
             n_tasks_per_batch=1, bin_masking=None, **kwargs):

    """Do binned optimization with scipy.optimize.minimize.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """

    optimizer = "minimize"
    optimizer_kwargs = dict(kwargs)

    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def differential_evolution(fun, binning_tuples, return_evals=False, 
                           optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
                           parallelization=None, max_processes=1, task_distribution="even", 
                           n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with scipy.optimize.differential_evolution as the optimizer.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
    """

    optimizer = "differential_evolution"
    optimizer_kwargs = dict(kwargs)

    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def basinhopping(fun, binning_tuples, return_evals=False, 
                 optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
                 parallelization=None, max_processes=1, task_distribution="even", 
                 n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with scipy.optimize.basinhopping as the optimizer.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
    """

    optimizer = "basinhopping"
    optimizer_kwargs = dict(kwargs)

    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def shgo(fun, binning_tuples, return_evals=False, 
         optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
         parallelization=None, max_processes=1, task_distribution="even", 
         n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with scipy.optimize.shgo as the optimizer.
    
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
    """

    optimizer = "shgo"
    optimizer_kwargs = dict(kwargs)

    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def dual_annealing(fun, binning_tuples, return_evals=False, 
                   optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2, 
                   parallelization=None, max_processes=1, task_distribution="even", 
                   n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with scipy.optimize.dual_annealing as the optimizer.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing
    """

    optimizer = "dual_annealing"
    optimizer_kwargs = dict(kwargs)

    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def direct(fun, binning_tuples, return_evals=False, 
           optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
           parallelization=None, max_processes=1, task_distribution="even",
           n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with scipy.optimize.direct as the optimizer.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html#scipy.optimize.direct
    """

    optimizer = "direct"
    optimizer_kwargs = dict(kwargs)
    
    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)



def iminuit(fun, binning_tuples, return_evals=False, 
           optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2,
           parallelization=None, max_processes=1, task_distribution="even",
           n_tasks_per_batch=1, bin_masking=None, **kwargs):
    """Do binned optimization with iminuit.minimize as the optimizer.

    See https://scikit-hep.org/iminuit/reference.html#scipy-like-interface
    """

    optimizer = "iminuit"
    optimizer_kwargs = dict(kwargs)
    
    return _run_optimizer(fun, binning_tuples, optimizer, optimizer_kwargs, return_evals, 
                          optima_comparison_rtol, optima_comparison_atol, parallelization,
                          max_processes, task_distribution, n_tasks_per_batch, bin_masking)

