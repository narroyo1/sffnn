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

    def __init__(
        self,
        z_samples_per_dimension,
        z_ranges_per_dimension,
        outer_level_scalar,
        outer_samples,
        device,
    ):
        # Create a tensor of samples on z-space.
        # [z0, z1, ... , zS]
        # dimensions: (z-samples mult, output dimensions)
        if not outer_samples:
            # Make a copy of the ranges to modify.
            z_ranges_per_dimension_cp = np.array(z_ranges_per_dimension)
            for z_range in z_ranges_per_dimension_cp:
                range_size = z_range[1] - z_range[0]
                # Shave off a percentage on each side.
                z_range[0] += range_size / 20.0
                z_range[1] -= range_size / 20.0
        self._z_samples_np = sample_uniform(
            z_ranges_per_dimension_cp, z_samples_per_dimension
        )
        self.z_samples_pt = to_tensor(self._z_samples_np, device)

        self.z_ranges_per_dimension = z_ranges_per_dimension
        # self.z_samples_per_dimension = z_samples_per_dimension
        self.z_dimensions = z_ranges_per_dimension.shape[0]
        self._outer_level_scalar = outer_level_scalar

        self.smaller_than_ratios = sample_uniform(
            np.array([[0.0, 1.0] for _ in range(self.z_dimensions)]),
            z_samples_per_dimension,
        )
        self.smaller_than_ratios_pt = to_tensor(self.smaller_than_ratios, device)

        self.less_scalar, self.more_scalar = self.calculate_scalars()
        # print("less_scalar", self.less_scalar)
        # print("more_scalar", self.more_scalar)

        # self.less_scalar_bias, self.more_scalar_bias = self.biased_scalars(
        #    self.less_scalar, self.more_scalar
        # )

    def calculate_scalars(self):
        """
        This method calculates the alpha and beta scalars for every z-sample.
        """

        z_samples_size = self._z_samples_np.shape[0]
        z_samples_dimensions = self._z_samples_np.shape[1]
        less_scalar = np.zeros((z_samples_size, z_samples_dimensions), dtype=float)
        more_scalar = np.zeros((z_samples_size, z_samples_dimensions), dtype=float)

        for idx in range(z_samples_size):
            for dim in range(z_samples_dimensions):
                min_val = self.z_ranges_per_dimension[dim][0]
                max_val = self.z_ranges_per_dimension[dim][1]
                a_n = self._z_samples_np[idx, dim] - min_val
                if a_n == 0.0:
                    less_scalar[idx, dim] = self._outer_level_scalar
                    more_scalar[idx, dim] = 0.0
                    continue
                b_n = max_val - self._z_samples_np[idx, dim]
                if b_n == 0.0:
                    more_scalar[idx, dim] = self._outer_level_scalar
                    less_scalar[idx, dim] = 0.0
                    continue
                alpha = 1.0 / (2.0 * a_n)
                beta = 1.0 / (2.0 * b_n)
                less_scalar[idx, dim] = alpha
                more_scalar[idx, dim] = beta

        normalizer = 1.0 / np.max(less_scalar)
        # print("normalizer:", normalizer)
        less_scalar *= normalizer
        more_scalar *= normalizer

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
