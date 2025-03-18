import numpy as np

def get_min_bins(bin_tuples, function_values, target_dims):
    """
    For each bin in the target dimensions, get the row index for the bin_tuples entry with the 
    minimum function value.

    Parameters:
    - bin_tuples (np.ndarray): An (N, d) array. Each row represents a tuple of per-dimension bin indices.
    - function_values (np.ndarray): A (N,) array of function values corresponding to each bin.
    - target_dims (list or tuple): Dimensions over which to retain the bins and minimize over the 
                                   remaining dimensions.

    Returns:
    - min_bin_row_indices (np.ndarray): For each bin in the target dimensions, this array contains 
                                        the row index for the bin_tuples entry with the minimum 
                                        function value.
    """

    # Make sure target_dims is a tuple
    if (type(target_dims) is int):
        target_dims = (target_dims,)
    target_dims = tuple(target_dims)

    # Extract the per-dimension bin indices for the target dimensions
    target_perdim_bin_indices = bin_tuples[:, target_dims]

    # Find unique combinations of per-dimension bin indices in the target dimensions
    unique_perdim_indices, inverse_indices = np.unique(target_perdim_bin_indices, axis=0, return_inverse=True)

    # Number of bins in the target dimensions
    num_target_bins = unique_perdim_indices.shape[0]
    
    # Prepare an array for the result
    min_bin_row_indices = np.empty(num_target_bins, dtype=int)

    # Now find the correct bin_tuples index for each bin in the target dimensions.
    for i in range(num_target_bins):
        # Create a mask that filters out elements relevant for bin i
        mask = np.where(inverse_indices == i)[0]
        # Conditional on this mask, pick out the index of the smallest function value
        fmin_index = mask[np.argmin(function_values[mask])]
        # Save this index
        min_bin_row_indices[i] = fmin_index

    return min_bin_row_indices

