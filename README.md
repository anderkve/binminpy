<br>
<div align="center">
<img src="logo.png" width="400"/>
</div>
<br>

# binminpy

A Python package for binned and parallelised function optimisation.


## Installation

Simply install using `pip`:

```terminal
pip install git+https://github.com/anderkve/binminpy.git
```

## Minimal example

Here is a minimal example of how to run `scipy.optimize.minimize` in parallel on a binned three-dimensional input space:

```python
import binminpy

# An example target function
def target_function(x):
    """ An n-dimensional version of the Himmelblau function."""
    result = 0.
    for i in range(len(x)-1):
        result += (x[i]**2 + x[i+1] - 11.)**2 + (x[i] + x[i+1]**2 - 7.)**2
    return result

# Example binning setup for a 3D input space using 5x5x2 bins.
binning_tuples = [(-6.0, 6.0, 5), (-6.0, 6.0, 5), (-6.0, 6.0, 2)]

# Run binminpy.minimize, parallelized with four processes
# using concurrent.futures.ProcessPoolExecutor (parallelization="ppe").
result = binminpy.minimize(
    target_function, 
    binning_tuples, 
    method="L-BFGS-B",
    tol=1e-9,
    return_evals=False,
    optima_comparison_rtol=1e-6, 
    optima_comparison_atol=1e-4,
    parallelization="ppe",
    max_processes=4,
)
```

## Parallelization

**binminpy** can be run in four different parallelization modes:

- serial: No parallelization
- PPE: Distribute tasks evenly across multiple processes using `ProcessPoolExecutor` from `concurrent.futures`
- MPI, even: Distribute tasks evenly across MPI processes using `mpi4py`.
- MPI, dynamic: Use a master-worker pattern to distribute tasks across MPI processes using `mpi4py`.