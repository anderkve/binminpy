"""
binminpy
========

A package for parallelised, binned function optimisation.
"""

__version__ = "0.1.0"
__author__ = "Anders Kvellestad"
__credits__ = ""



def minimize(fun, binning_tuples, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None, return_evals=False, 
             optima_comparison_rtol=1e-6, optima_comparison_atol=1e-2):
    """Function with a similar interface as scipy.optimize.minimize.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """

    from binminpy.BinnedOptimizer import BinnedOptimizer

    optimizer = "minimize"
    optimizer_kwargs = {
        "args": args,
        "hess": hess,
        "hessp": hessp,
        "bounds": bounds,
        "constraints": constraints,
        "tol": tol,
        "callback": callback,
        "options": options,
    }

    binned_opt = BinnedOptimizer(
        fun,
        binning_tuples,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        return_evals=return_evals,
        optima_comparison_rtol=optima_comparison_rtol,
        optima_comparison_atol=optima_comparison_atol
    )

    output = binned_opt.run()

    return output
