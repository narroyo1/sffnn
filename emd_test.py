"""
This module contains class EMDTest.
"""

import numpy as np

import torch

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils import sample_random, to_tensor

# The number of points in X to test for goal 1.
EMD_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
# This number cannot be too big since EMD is expensive to calculate.
EMD_SAMPLES_PER_TEST_POINT = 500


class EMDTest:
    """
    This class implements the EMD (Earth Movers Distance) test on the model.
    """

    def __init__(self, z_samples, datasets, plotter, writer, model, device):
        self.plotter = plotter
        self.writer = writer
        self.model = model
        self.device = device
        self.z_ranges_per_dimension = z_samples.z_ranges_per_dimension
        #self.z_space_size = z_samples.Z_SPACE_SIZE

        self.y_test = datasets.y_test
        self.x_test = datasets.x_test

        self.x_test_pt = to_tensor(datasets.x_test, device)

        self.x_orderings_pt = [
            torch.sort(self.x_test_pt[:, i])[1] for i in range(self.x_test_pt.shape[1])
        ]
        self.x_orderings_np = [
            np.argsort(self.x_test[:, i]) for i in range(self.x_test.shape[1])
        ]

    def calculate_emd(self, y_pred_d):
        """
        This function calculates the emd (Earth Mover's Distance) between a model prediction and
        the test data set. Calculating emd is very expensive (O(n^2)) so in order to speed up the
        calculation, the test and prediciton data points are separated into groups and their emd's
        are averaged together.
        """

        test_point_spacing = int(self.y_test.shape[0] / EMD_TEST_POINTS)

        num_dimensions = self.x_test.shape[1]
        mean_emds = np.zeros(num_dimensions)
        for dimension in range(num_dimensions):
            local_emds = np.zeros(EMD_TEST_POINTS + 2)
            x_np = np.zeros(EMD_TEST_POINTS + 2)
            for point in range(EMD_TEST_POINTS):
                start = point * test_point_spacing - EMD_SAMPLES_PER_TEST_POINT / 2
                stop = start + EMD_SAMPLES_PER_TEST_POINT
                start = int(max(start, 0))
                stop = int(stop)

                # Calculate the distances between every point in one set and every point in
                # the other.
                distances = cdist(
                    self.y_test[self.x_orderings_np[dimension]][start:stop],
                    y_pred_d[self.x_orderings_np[dimension]][start:stop],
                )

                # Calculate the point to point matching the minimizes the EMD.
                assignment = linear_sum_assignment(distances)
                local_emds[point + 1] = distances[assignment].sum() / (stop - start)
                x_np[point + 1] = np.mean(
                    self.x_test[self.x_orderings_np[dimension]][start:stop][
                        :, dimension
                    ]
                )

            mean_emds[dimension] = np.mean(local_emds)
            x_np[0] = self.x_test[self.x_orderings_pt[dimension][0]][dimension]
            local_emds[0] = local_emds[1]
            x_np[-1] = self.x_test[self.x_orderings_pt[dimension][-1]][dimension]
            local_emds[-1] = local_emds[-2]

            self.plotter.plot_emd(x_np=x_np, local_emds=local_emds, dimension=dimension)

        mean_emd = np.mean(mean_emds)
        return mean_emd

    def step(self, epoch):
        """
        Runs and plots a step of the EMD test.
        """

        # With grad off make a prediction over a random set of z-samples and the test x
        # data points.
        with torch.no_grad():
            test_size = self.x_test_pt.shape[0]
            z_test = sample_random(self.z_range, test_size, self.z_space_size)
            z_test_pt = to_tensor(z_test, self.device)
            y_pred = self.model.forward_z(self.x_test_pt, z_test_pt)

        # Create a numpy version of the prediction tensor.
        y_pred_d = y_pred.cpu().detach()

        # First test: calculate the emd.
        mean_emd = self.calculate_emd(y_pred_d)

        self.plotter.plot_datasets_preds(y_pred_d)

        self.writer.log_emd(mean_emd, epoch)
