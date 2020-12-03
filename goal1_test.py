"""
This module contains class Goal1Test.
"""
# pylint: disable=no-member

import numpy as np

import torch

from utils import to_tensor

# The number of points in X to test for goal 1.
GOAL1_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
GOAL1_SAMPLES_PER_TEST_POINT = 800


class Goal1Test:
    """
    This class implements the goal 1 test on the model. That is it checks that
    the predictions for the z-samples have the right distribution ratios.
    """

    def __init__(self, z_samples, datasets, plotter, writer, device):
        self.plotter = plotter
        self.writer = writer
        self.device = device

        self.z_samples = z_samples
        # self.less_than_ratios = z_samples.less_than_ratios

        self.y_test = datasets.y_test
        self.x_test = datasets.x_test

        self.y_test_pt = to_tensor(datasets.y_test, device)
        self.x_test_pt = to_tensor(datasets.x_test, device)
        self.x_orderings_pt = [
            torch.sort(self.x_test_pt[:, i])[1] for i in range(self.x_test_pt.shape[1])
        ]

    def test_goal1(self, y_predict_mat):
        """
        This method tests the hypothesis that every z-line divides the level by half.
        """
        from trainer import get_movement_scalars, get_unit_and_mag, get_direction_slots

        differences = self.y_test_pt - y_predict_mat
        D, _ = get_unit_and_mag(differences)

        D1, D2 = get_direction_slots(D, self.device)

        w_bp = get_movement_scalars(D1, D2, self.z_samples)

        total_movement = torch.sum(D * w_bp.unsqueeze(2), dim=1)
        d, l = get_unit_and_mag(total_movement)

        # This matrix tells for every test data point if it is smaller than each
        # z-sample prediction.
        # dimensions: (z-samples, test datapoints)
        # differences[differences < 0] = float("inf")
        squared = differences ** 2
        summation = torch.sum(squared, dim=2)
        closest, indices = torch.min(summation, axis=0)
        # closest = differences[closest, torch.arange(differences.shape[1])]
        c = torch.bincount(indices)

        return d.cpu().numpy(), l.cpu().numpy(), c.cpu().numpy()

        smaller_than = torch.le(self.y_test_pt.squeeze(), y_predict_mat) + 0.0

        # This is the number of test points separting group middle points.
        test_point_spacing = int(self.y_test.shape[0] / GOAL1_TEST_POINTS)

        # This is the mean ratio of smaller than over all test data points for each z-sample.
        # dimensions: (z-samples)
        goal1_mean = torch.mean(smaller_than, dim=1,)
        # This is the error for every mean ratio above.
        # dimensions: (z-samples)
        goal1_err = goal1_mean - self.less_than_ratios
        # This is the absolute error for every z-sample. It has to be absolute so that
        # they doesn't cancel each other when averaging.
        # dimensions: (z-samples)
        goal1_err_abs = torch.abs(goal1_err)
        # This is the single mean value of the absolute error of all z-samples.
        goal1_mean_err_abs = torch.mean(goal1_err_abs)

        num_dimensions = self.x_test.shape[1]
        for dimension in range(num_dimensions):
            # This array will have the goal error means for every test point's vicinity.
            # Add 2 members for the plot edges.
            local_goal1_err = np.zeros(GOAL1_TEST_POINTS + 2)
            local_goal1_max_err = np.zeros(GOAL1_TEST_POINTS + 2)
            # dimensions: (z-samples, test datapoint groups)
            local_goal1_err_zsample = np.zeros(
                (y_predict_mat.shape[0], GOAL1_TEST_POINTS + 2)
            )

            x_np = np.zeros(GOAL1_TEST_POINTS + 2)
            for point in range(GOAL1_TEST_POINTS):
                # For the current test point, select the start and stop indexes.
                start = point * test_point_spacing - GOAL1_SAMPLES_PER_TEST_POINT / 2
                stop = start + GOAL1_SAMPLES_PER_TEST_POINT
                start = int(max(start, 0))
                stop = int(stop)

                # This is the mean ratio of smaller than over all the data points in this
                # vicinity.
                # dimension: (z-samples)
                smaller_than_mean = torch.mean(
                    smaller_than[:, self.x_orderings_pt[dimension]][:, start:stop],
                    dim=1,
                )

                # Get the error by substracting it from the expected ratio and calculate
                # the absolute value.
                # dimension: (z-samples)
                smaller_than_mean_abs = torch.abs(
                    smaller_than_mean - self.less_than_ratios
                )

                local_goal1_err[point + 1] = torch.mean(smaller_than_mean_abs, dim=0)
                local_goal1_err_zsample[
                    :, point + 1
                ] = smaller_than_mean_abs.cpu().numpy()
                local_goal1_max_err[point + 1] = torch.max(smaller_than_mean_abs)
                # Calculate the x value for the plot as the average of all data points considered.
                x_np[point + 1] = torch.mean(
                    self.x_test_pt[self.x_orderings_pt[dimension]][start:stop][
                        :, dimension
                    ]
                )

            # mean_goal1_errs[dimension] = np.mean(local_goal1_err)
            x_np[0] = self.x_test[self.x_orderings_pt[dimension][0]][dimension]
            local_goal1_err[0] = local_goal1_err[1]
            local_goal1_max_err[0] = local_goal1_max_err[1]
            x_np[-1] = self.x_test[self.x_orderings_pt[dimension][-1]][dimension]
            local_goal1_err[-1] = local_goal1_err[-2]
            local_goal1_max_err[-1] = local_goal1_max_err[-2]

            self.plotter.plot_goal1(
                x_np=x_np,
                local_goal1_err=local_goal1_err,
                global_goal1_err=goal1_mean_err_abs,
                dimension=dimension,
                local_goal1_err_zsample=local_goal1_err_zsample,
            )

        return goal1_mean_err_abs

    def step(self, epoch, y_predict_mat):
        """
        Runs and plots a step of the goal 1 test.
        """

        # Second test: Test training goal 1.
        return self.test_goal1(y_predict_mat)

        # self.writer.log_goal1_error(mean_goal1, epoch)
