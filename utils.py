"""
This module contains misc utility functions.
"""
# %%
from math import ceil

import numpy as np
import torch

# %%
def sample_uniform(ranges, sizes):
    """
    This function returns a uniformly distributed sample (grid) in a space
    of dims dimensions.
    @return nparray with shape (sizes, dims)
    """
    dimensions = ranges.shape[0]

    slices = []
    for dimension in range(dimensions):
        start = ranges[dimension, 0]
        end = ranges[dimension, 1]
        step = (end - start) / (sizes[dimension] - 1)
        slices.append(slice(start, end + 0.000000001, step))
    slices = tuple(slices)

    # Create a tensor with the grid coordinates.
    mgrid = np.mgrid[slices].reshape(dimensions, -1).T
    # Return a truncated tensor containing only the requested sizes.
    return mgrid


# print(sample_uniform(np.array([[1, 9]]), np.array([10])))
# print(sample_uniform(np.array([[1, 9], [0, 1]]), np.array([3, 3])))


# %%
def sample_random(ranges, sizes):
    """
    @return nparray with shape (size, dims)
    """
    dims = ranges.shape[0]
    start = ranges[:, 0]
    end = ranges[:, 1]
    samples = (np.random.rand(sizes, dims) * (end - start)) + start

    return samples


# print(sample_random((0.0, 1.0), 10, 1))
# print(sample_random((0.0, 1.0), 10, 2))

# %%
def to_tensor(nparray, device):
    """
    This fuction transforms a numpy array into a tensorflow tensor.
    """
    return torch.tensor(nparray, dtype=torch.float32).to(device=device)


# %%
