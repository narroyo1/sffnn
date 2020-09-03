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
    # Take the dimensions'th root of dimensions. That is the number of points on each dimension.
    # dim_size = sizes ** (1 / dimensions)
    # Round it to the nearest greater integer.
    # dim_size = np.ceil(dim_size)

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
def sample_random(range_, size, dims):
    """
    @return nparray with shape (size, dims)
    """
    start = range_[:, 0]
    end = range_[:, 1]
    samples = (np.random.rand(size, dims) * (end - start)) + start

    return samples


# print(sample_random((0.0, 1.0), 10, 1))
# print(sample_random((0.0, 1.0), 10, 2))
# %%


def create_z_samples(z_range, total_levels, z_space_size):
    """
    This function creates the set of z-samples.
    """
    inner_lines = 2 ** total_levels - 1
    outer_lines = 2
    z_samples_size = inner_lines + outer_lines
    z_samples = sample_uniform(z_range, z_samples_size, z_space_size)

    return z_samples


# create_z_samples((1.0, 2.0), 0)

# %%
def to_tensor(nparray, device):
    """
    This fuction transforms a numpy array into a tensorflow tensor.
    """
    return torch.tensor(nparray, dtype=torch.float32).to(device=device)


# %%

# pylint: disable=wrong-import-position, wrong-import-order
from collections import defaultdict
from datetime import datetime, timedelta

timer_aggregates = defaultdict(timedelta)


def timer(func):
    """
    Timer utility function to track down bottlenecks.
    """

    def func_wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        funcname = func.__name__
        timer_aggregates[funcname] += datetime.now() - start
        print("Accumulated time for {} {}".format(funcname, timer_aggregates[funcname]))

    return func_wrapper


# %%
