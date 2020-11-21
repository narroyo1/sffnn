"""
This module contains class ZSamples.
"""
# pylint: disable=bad-continuation

import numpy as np

from utils import sample_uniform, sample_random, to_tensor


class ZSamples:
    """
    This class encapsulates the z-samples set.
    """

    def __init__(self, experiment, device):

        # Create a tensor of samples on z-space.
        # [z0, z1, ... , zS]
        self.z_samples_radio = experiment["z_samples_radio"]
        self.z_samples_dimensions = experiment["z_samples_dimensions"]
        self.device = device

        if "z_samples_per_dimension" in experiment:
            z_ranges_per_dimension = self.get_ranges_per_dimension(
                self.z_samples_radio, self.z_samples_dimensions
            )
            z_samples_per_dimension = experiment["z_samples_per_dimension"]
            z_samples = sample_uniform(z_ranges_per_dimension, z_samples_per_dimension)
            z_samples = self.rescale_samples(z_samples, self.z_samples_radio)
            # z_samples = self.clip_samples(z_samples, self.z_samples_radio)
        else:
            z_samples = experiment["z_samples"]

        self.z_samples = to_tensor(z_samples, device)
        self.z_sample_labels = experiment.get(
            "z_sample_labels",
            ["$z_{{{}}}$".format(i) for i in range(z_samples.shape[0])],
        )

        self.outer_level_scalar = experiment.get("outer_level_scalar")

    @staticmethod
    def get_ranges_per_dimension(z_samples_radio, dimensions):
        """
        This method gets a ranges array to be used with the sample free functions.
        """
        z_ranges_per_dimension = np.zeros((dimensions, 2))
        z_ranges_per_dimension[:, 0] = -z_samples_radio
        z_ranges_per_dimension[:, 1] = z_samples_radio

        return z_ranges_per_dimension

    @staticmethod
    def clip_samples(samples, z_samples_radio):
        in_hypersphere = (
            np.sum(samples * samples, axis=1) <= z_samples_radio * z_samples_radio
        )
        return samples[in_hypersphere]

    @staticmethod
    def rescale_samples(samples, z_samples_radio):
        out_of_hypersphere = (
            np.sum(samples * samples, axis=1) > z_samples_radio * z_samples_radio
        )

        length = samples[out_of_hypersphere] * samples[out_of_hypersphere]
        length = np.sum(length, axis=1)
        length = np.sqrt(length)
        length = samples[out_of_hypersphere] / length[..., np.newaxis]

        samples[out_of_hypersphere] = length * z_samples_radio * 0.98

        return samples

    def sample_random(self, size):
        """
        This method gets size random elements in Z.
        """
        z_ranges_per_dimension = self.get_ranges_per_dimension(
            self.z_samples_radio, self.z_samples_dimensions
        )
        z_samples = sample_random(z_ranges_per_dimension, size * 2)
        z_samples = self.clip_samples(z_samples, self.z_samples_radio)
        assert z_samples.shape[0] >= size

        return z_samples[:size]
