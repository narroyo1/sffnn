"""
This module contains class EMDTest.
"""
# pylint: disable=no-member

import numpy as np

import torch

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils import to_tensor

# The number of points in X to test for goal 1.
EMD_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
# This number cannot be too big since EMD is expensive to calculate.
EMD_SAMPLES_PER_TEST_POINT = 500


class EMDTest:
    """
    This class implements the EMD (Earth Movers Distance) test on the model.
    """

    def __init__(self, datasets, device):

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

        local_emds = []
        num_dimensions = self.x_test.shape[1]
        mean_emds = np.zeros(num_dimensions)
        for dimension in range(num_dimensions):
            local_emd = np.zeros(EMD_TEST_POINTS + 2)
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
                local_emd[point + 1] = distances[assignment].sum() / (stop - start)
                x_np[point + 1] = np.mean(
                    self.x_test[self.x_orderings_np[dimension]][start:stop][
                        :, dimension
                    ]
                )

            mean_emds[dimension] = np.mean(local_emd)
            x_np[0] = self.x_test[self.x_orderings_pt[dimension][0]][dimension]
            local_emd[0] = local_emd[1]
            x_np[-1] = self.x_test[self.x_orderings_pt[dimension][-1]][dimension]
            local_emd[-1] = local_emd[-2]

            local_emds.append((x_np, local_emd))

        mean_emd = np.mean(mean_emds)
        return mean_emd, local_emds

    def step(self, y_pred_d):
        """
        Runs and plots a step of the EMD test.
        """
        # return

        # First test: calculate the emd.
        mean_emd, local_emds = self.calculate_emd(y_pred_d)

        return mean_emd, local_emds
