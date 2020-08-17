"""
This module contains class ZSamples.
"""

import numpy as np

from utils import sample_uniform, to_tensor


class ZSamples:
    """
    This class encapsulates the z-samples set.
    """

    # Currently Z space is only supported to be 1.
    # Z_SPACE_SIZE = 1

    # If this value is 1.0 it may pull the outer z-lines too far and affect
    # the z-lines immediately next to them.
    OUTER_LEVEL_SCALAR = 0.2  # exp 1, 2, 3, 4, 5
    # OUTER_LEVEL_SCALAR = 0.1  # exp 6

    def __init__(self, num_z_samples, z_range, device):
        # Create a tensor of samples on z-space.
        # [z0, z1, ... , zS]
        z_samples = sample_uniform(z_range, num_z_samples)
        self.z_samples = to_tensor(z_samples, device)

        self.z_range = z_range

        z_space_size = z_range.shape[0]
        self.less_scalar = []
        self.more_scalar = []
        for dimension in range(z_space_size):
            z_samples_dim = sample_uniform(
                z_range[dimension : dimension + 1],
                num_z_samples[dimension : dimension + 1],
            )
            less_scalar, more_scalar = self.calculate_scalars(z_samples_dim)
            self.less_scalar.append(less_scalar)
            self.more_scalar.append(more_scalar)
        # print("less_scalar", self.less_scalar)
        # print("more_scalar", self.more_scalar)

        self.less_scalar_bias, self.more_scalar_bias = self.biased_scalars(
            self.less_scalar, self.more_scalar
        )

    @staticmethod
    def calculate_scalars(z_samples):
        """
        This method calculates the alpha and beta scalars for every z-sample.
        """

        num_z_samples = z_samples.shape[0]
        less_scalar = np.zeros((num_z_samples,), dtype=float)
        more_scalar = np.zeros((num_z_samples,), dtype=float)
        min_val = z_samples[0]
        max_val = z_samples[-1]

        for idx in range(1, num_z_samples - 1):
            a_n = z_samples[idx] - min_val
            b_n = max_val - z_samples[idx]
            alpha = 1.0 / (2.0 * a_n)
            beta = 1.0 / (2.0 * b_n)
            less_scalar[idx] = alpha
            more_scalar[idx] = beta

        normalizer = 1.0 / np.max(less_scalar)
        # print("normalizer:", normalizer)
        less_scalar *= normalizer
        more_scalar *= normalizer

        less_scalar[0] = more_scalar[-1] = ZSamples.OUTER_LEVEL_SCALAR
        less_scalar[-1] = more_scalar[0] = 0.0

        return less_scalar, more_scalar

    @staticmethod
    def biased_scalars(less_scalar, more_scalar):
        """ Use binary traversal to create the labels. """

        def biased_scalars_helper(start, end, scalars, level=0):
            if start > end:
                return

            if (end - start) % 2 == 0:
                mid1 = mid2 = int((end - start) / 2) + start
            else:
                mid1 = int((end - start) / 2) + start
                mid2 = int((end - start) / 2) + start + 1

            indexes = set([mid1, mid2, start, end])
            for index in indexes:
                scalars[index] /= 2 ** level
            # if start != end:
            #    scalars[start] *= 1.2
            #    scalars[end] *= 1.2

            biased_scalars_helper(start + 1, mid1 - 1, scalars, level + 1)
            biased_scalars_helper(mid2 + 1, end - 1, scalars, level + 1)

        less_scalar_bias = np.copy(less_scalar)
        more_scalar_bias = np.copy(more_scalar)
        # This step is important to prevent group fluctuation.
        biased_scalars_helper(1, len(less_scalar_bias) - 2, less_scalar_bias)
        biased_scalars_helper(1, len(more_scalar_bias) - 2, more_scalar_bias)
        # print("less_scalar_bias", self.less_scalar_bias)
        # print("more_scalar_bias", self.more_scalar_bias)

        return less_scalar_bias, more_scalar_bias
