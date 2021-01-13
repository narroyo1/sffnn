"""
This module contains class ZSamples.
"""
# pylint: disable=bad-continuation

import numpy as np

import torch

from utils import sample_uniform, sample_random, to_tensor


def sample_hypersphere(z_ranges, z_samples):
    """
    This function will return multiple samples with coordinates 0, 0. This is by
    design, since having only one sample at 0, 0 would leave it in an unfair competition
    and would be learned more slowly.
    """
    angle = np.linspace(-np.pi, np.pi, z_samples[0] + 1)
    angle = angle[1:]
    line = np.sqrt(np.linspace(0.0, 1.0, z_samples[1])) * z_ranges[0][1]
    # line = np.linspace(0.0, 1.0, z_samples[1] + 1) * z_ranges[0][1]
    # line = line[1:]
    # line = line[:-1]
    # line[-1] = 9.5

    a = (line[:, np.newaxis] * np.cos(angle)).flatten()
    b = (line[:, np.newaxis] * np.sin(angle)).flatten()

    # return np.append(np.column_stack((a, b)), [[0, 0]], axis=0)
    # return (
    #    np.column_stack((a, b)),
    #    np.tile(np.arange(z_samples[1]), z_samples[0]).flatten(),
    # )
    return np.column_stack((a, b))


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
            # z_samples = sample_hypersphere(
            #    z_ranges_per_dimension, z_samples_per_dimension
            # )
            z_sample_spacing = (
                self.z_samples_radio * 2 / (z_samples_per_dimension[0] - 1)
            )
            z_samples = self.clip_samples(z_samples, self.z_samples_radio)
            # z_samples = self.rescale_samples(z_samples, self.z_samples_radio)
            radios = self.calculate_radios(
                z_samples, self.z_samples_radio, z_sample_spacing
            )
            self.z_sample_spacing = z_sample_spacing
            outer_level = self.get_outer_level(z_samples, self.z_samples_radio)
        else:
            z_samples = experiment["z_samples"]

        self.samples = to_tensor(z_samples, device)
        self.radios = to_tensor(radios, device)
        # self.ring_numbers = z_samples_per_dimension[1]
        self.outer_level = torch.tensor(outer_level, dtype=torch.bool).to(device=device)
        # self.outer_samples = to_tensor(outer_samples, device)
        self.labels = experiment.get(
            "z_sample_labels",
            ["$z_{{{}}}$".format(i) for i in range(z_samples.shape[0])],
        )
        self.selection_size = int(
            z_samples.shape[0] * experiment.get("selection_size", 1.0)
        )

        self.outer_level_scalar = experiment.get("outer_level_scalar")

    @staticmethod
    def calculate_radios(z_samples, z_samples_radio, z_sample_spacing):
        distance_from_center = np.sqrt(np.sum(z_samples ** 2, axis=1))
        distance_to_circle = z_samples_radio - distance_from_center
        radios = np.minimum(z_sample_spacing / 2.0, distance_to_circle)
        # radios = np.minimum(z_sample_spacing * 1.0, distance_to_circle)

        return radios

    def selection(self):
        # return self.samples, self.outer_level
        import torch

        indices = np.random.choice(
            self.samples.shape[0], self.selection_size, replace=False
        )
        indices = torch.tensor(indices, dtype=torch.long).to(self.device)
        indices = torch.sort(indices).values

        return self.samples[indices, :], self.outer_level[indices]

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

        in_hypersphere = np.sum(samples ** 2, axis=1) <= z_samples_radio ** 2

        return samples[in_hypersphere]

    @staticmethod
    def get_outer_level(samples, z_samples_radio):

        EPSILON = np.finfo(np.float32).eps * 500
        outer_level = np.sum(samples ** 2, axis=1) - z_samples_radio ** 2
        outer_level = np.abs(outer_level) < EPSILON

        return outer_level

    @staticmethod
    def rescale_samples(samples, z_samples_radio):
        out_of_hypersphere = np.sum(samples ** 2, axis=1) > z_samples_radio ** 2

        length = samples[out_of_hypersphere] ** 2
        length = np.sum(length, axis=1)
        length = np.sqrt(length)
        length = samples[out_of_hypersphere] / length[..., np.newaxis]

        samples[out_of_hypersphere] = length * z_samples_radio  # * 0.95

        return samples  # , out_of_hypersphere

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
