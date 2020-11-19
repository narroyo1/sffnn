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

    def __init__(self, experiment, device):

        # Create a tensor of samples on z-space.
        # [z0, z1, ... , zS]
        self.z_samples_radio = experiment["z_samples_radio"]
        # self.z_ranges_per_dimension = z_ranges_per_dimension
        self.device = device

        if "z_samples_per_dimension" in experiment:
            z_samples_per_dimension = experiment["z_samples_per_dimension"]
            z_ranges_per_dimension = np.zeros((z_samples_per_dimension.shape[0], 2))
            z_ranges_per_dimension[:, 0] = -self.z_samples_radio
            z_ranges_per_dimension[:, 1] = self.z_samples_radio
            z_samples = sample_uniform(z_ranges_per_dimension, z_samples_per_dimension)
            in_hypersphere = (
                np.sum(z_samples * z_samples, axis=1)
                <= self.z_samples_radio * self.z_samples_radio
            )
            z_samples = z_samples[in_hypersphere]
        else:
            z_samples = experiment["z_samples"]
        self.z_samples = to_tensor(z_samples, device)
        self.z_sample_labels = experiment.get(
            "z_sample_labels",
            ["$z_{{{}}}$".format(i) for i in range(z_samples.shape[0])],
        )

        # self.less_than_ratios = self.calculate_ratios()

        self.outer_level_scalar = experiment.get("outer_level_scalar")

        # self.less_scalar, self.more_scalar = self.calculate_scalars()

    def calculate_ratios(self):
        """
        This method calculates the smaller than ratios array.
        """

        min_val = self.z_range[0][0]
        max_val = self.z_range[0][1]
        size = max_val - min_val

        return to_tensor(
            [(z_sample - min_val) / size for z_sample in self.z_samples.squeeze()],
            self.device,
        )

    def calculate_scalars(self):
        """
        This method calculates the alpha and beta scalars for every z-sample.
        """

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
=======
        z_samples_size = self.z_samples.shape[0]
        less_scalar = np.zeros((z_samples_size,), dtype=float)
        more_scalar = np.zeros((z_samples_size,), dtype=float)
        min_val = self.z_range[0][0]
        max_val = self.z_range[0][1]

        for idx in range(z_samples_size):
            a_n = self.z_samples[idx] - min_val
            if a_n == 0.0 and self.outer_level_scalar is not None:
                less_scalar[idx] = self.outer_level_scalar
                more_scalar[idx] = 0.0
                continue
            b_n = max_val - self.z_samples[idx]
            if b_n == 0.0 and self.outer_level_scalar is not None:
                more_scalar[idx] = self.outer_level_scalar
                less_scalar[idx] = 0.0
                continue
            alpha = 1.0 / (2.0 * a_n)
            beta = 1.0 / (2.0 * b_n)
            less_scalar[idx] = alpha
            more_scalar[idx] = beta
>>>>>>> master

        normalizer = 1.0 / np.max(less_scalar)
        # print("normalizer:", normalizer)
        less_scalar *= normalizer
        more_scalar *= normalizer

        return less_scalar, more_scalar
        """

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
