"""
This module contains class ZSamples.
"""
# pylint: disable=bad-continuation

import numpy as np

from utils import sample_uniform, to_tensor


class ZSamples:
    """
    This class encapsulates the z-samples set.
    """

    # Currently Z space is only supported to be 1.
    # Z_SPACE_SIZE = 1

    def __init__(
        self,
        z_samples_per_dimension,
        z_ranges_per_dimension,
        outer_level_scalar,
        device,
    ):
        # Create a tensor of samples on z-space.
        # [z0, z1, ... , zS]
        # dimensions: (z-samples mult, output dimensions)
        self.z_samples_np = sample_uniform(
            z_ranges_per_dimension, z_samples_per_dimension
        )
        self.z_samples_pt = to_tensor(self.z_samples_np, device)

        self.z_ranges = z_ranges_per_dimension
        self.outer_level_scalar = outer_level_scalar

        # z_space_size = z_range.shape[0]
        # self.less_scalar = []
        # self.more_scalar = []
        # for dimension in range(z_space_size):
        #    z_samples_dim = sample_uniform(
        #        z_range[dimension : dimension + 1],
        #        z_samples_per_dimension[dimension : dimension + 1],
        #    )
        #    less_scalar, more_scalar = self.calculate_scalars(z_samples_dim)
        #    self.less_scalar.append(less_scalar)
        #    self.more_scalar.append(more_scalar)

        self.less_scalar, self.more_scalar = self.calculate_scalars()
        # print("less_scalar", self.less_scalar)
        # print("more_scalar", self.more_scalar)

        self.less_scalar_bias, self.more_scalar_bias = self.biased_scalars(
            self.less_scalar, self.more_scalar
        )

    def calculate_scalars(self):
        """
        This method calculates the alpha and beta scalars for every z-sample.
        """

        z_samples_per_dimension = self.z_samples_np.shape[0]
        z_samples_dimensions = self.z_samples_np.shape[1]
        less_scalar = np.zeros(
            (z_samples_per_dimension, z_samples_dimensions), dtype=float
        )
        more_scalar = np.zeros(
            (z_samples_per_dimension, z_samples_dimensions), dtype=float
        )
        min_val = self.z_samples_np[0]
        max_val = self.z_samples_np[-1]

        for idx in range(1, z_samples_per_dimension - 1):
            a_n = self.z_samples_np[idx] - min_val
            b_n = max_val - self.z_samples_np[idx]
            alpha = 1.0 / (2.0 * a_n)
            beta = 1.0 / (2.0 * b_n)
            less_scalar[idx] = alpha
            more_scalar[idx] = beta

        normalizer = 1.0 / np.max(less_scalar)
        # print("normalizer:", normalizer)
        less_scalar *= normalizer
        more_scalar *= normalizer

        less_scalar[0] = more_scalar[-1] = self.outer_level_scalar
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
