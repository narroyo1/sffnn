"""
This module contains class Tester.
"""

import numpy as np

import torch

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from plotter import Plotter
from writer import Writer
from utils import sample_random, to_tensor

# pylint: disable=bad-continuation

# The number of epochs skipped before running the tests.
SKIP_EPOCHS = 5  # exp 1, 2, 3, 4, 5
# SKIP_EPOCHS = 10  # exp 6

# The number of points in X to test for goal 1.
EMD_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
# This number cannot be too big since EMD is expensive to calculate.
EMD_SAMPLES_PER_TEST_POINT = 500

# The number of points in X to test for goal 1.
GOAL1_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
GOAL1_SAMPLES_PER_TEST_POINT = 800  # exps 1, 2, 3, 4, 5, 6


class Tester:
    """
    This class implements a mechanism to test the performance of a neural network
    that produces stochastic outputs.
    """

    def __init__(
        self, z_samples, datasets, trainer, model, device, **kwargs,
    ):
        self.model = model
        self.device = device
        self.z_samples = z_samples.z_samples
        self.z_range = z_samples.z_range
        self.z_space_size = z_samples.Z_SPACE_SIZE
        self.z_samples_size = self.z_samples.shape[0]

        self.smaller_than_ratios = np.linspace(0.0, 1.0, self.z_samples_size)
        self.smaller_than_ratios = to_tensor(self.smaller_than_ratios, device)

        self.y_test = datasets.y_test
        self.x_test = datasets.x_test
        # self.x_range_test = datasets.x_range_test
        self.x_space_size = len(datasets.x_dimensions)

        self.y_test_pt = to_tensor(datasets.y_test, device)
        self.x_test_pt = to_tensor(datasets.x_test, device)
        self.x_orderings_pt = [
            torch.sort(self.x_test_pt[:, i])[1] for i in range(self.x_test_pt.shape[1])
        ]
        self.x_orderings_np = [
            np.argsort(self.x_test[:, i]) for i in range(self.x_test.shape[1])
        ]

        self.writer = Writer(
            datasets_target_function=datasets.target_function_desc,
            trainer_params=trainer.params_desc,
            datasets_params=datasets.params_desc,
        )

        self.plotter = Plotter(
            datasets=datasets, z_samples_size=self.z_samples_size, **kwargs,
        )

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

    def test_goal1(self, y_predict_mat, mon_incr):
        """
        This method tests the hypothesis that every z-line divides the level by half.
        """

        # This matrix tells for every test data point if it is smaller than each
        # z-sample prediction.
        smaller_than = torch.le(self.y_test_pt.squeeze(), y_predict_mat) + 0.0

        test_point_spacing = int(self.y_test.shape[0] / GOAL1_TEST_POINTS)

        goal1_mean = torch.mean(smaller_than, dim=1,)
        goal1_err = goal1_mean - self.smaller_than_ratios
        goal1_err_abs = torch.abs(goal1_err)
        goal1_mean_err_abs = torch.mean(goal1_err_abs)

        num_dimensions = self.x_test.shape[1]
        # mean_goal1_errs = np.zeros(num_dimensions)
        for dimension in range(num_dimensions):
            local_goal1_err = np.zeros(GOAL1_TEST_POINTS + 2)
            x_np = np.zeros(GOAL1_TEST_POINTS + 2)
            for point in range(GOAL1_TEST_POINTS):
                start = point * test_point_spacing - GOAL1_SAMPLES_PER_TEST_POINT / 2
                stop = start + GOAL1_SAMPLES_PER_TEST_POINT
                start = int(max(start, 0))
                stop = int(stop)
                smaller_than_mean = torch.mean(
                    smaller_than[:, self.x_orderings_pt[dimension]][:, start:stop],
                    dim=1,
                )
                smaller_than_mean_abs = torch.abs(
                    smaller_than_mean - self.smaller_than_ratios
                )
                local_goal1_err[point + 1] = torch.mean(smaller_than_mean_abs, dim=0)
                x_np[point + 1] = torch.mean(
                    self.x_test_pt[self.x_orderings_pt[dimension]][start:stop][
                        :, dimension
                    ]
                )

            # mean_goal1_errs[dimension] = np.mean(local_goal1_err)
            x_np[0] = self.x_test[self.x_orderings_pt[dimension][0]][dimension]
            local_goal1_err[0] = local_goal1_err[1]
            x_np[-1] = self.x_test[self.x_orderings_pt[dimension][-1]][dimension]
            local_goal1_err[-1] = local_goal1_err[-2]

            self.plotter.plot_goals(
                x_np=x_np,
                local_goal1_err=local_goal1_err,
                global_goal1_err=goal1_mean_err_abs,
                mon_incr=mon_incr,
                dimension=dimension,
            )

        # mean_goal1 = np.mean(mean_goal1_errs)
        return goal1_mean_err_abs

    def test_goal2(self):

        x_goal = to_tensor(
            self.x_test[np.random.choice(self.x_test.shape[0], 10)], self.device
        )
        # x_goal = to_tensor(
        #    sample_random(self.x_range_test, 10, self.x_space_size), self.device,
        # )
        z_goal = torch.sort(
            to_tensor(sample_random(self.z_range, 15, self.z_space_size), self.device),
            dim=0,
        )[0]
        # Get the z-sample predictions for every test data point.
        y_predict_mat = self.model.get_z_sample_preds(x=x_goal, z_samples=z_goal,)

        y_predict_mat_d = y_predict_mat.cpu().detach().numpy()

        ascending = np.all(y_predict_mat_d[:-1] <= y_predict_mat_d[1:])
        return ascending

    def step(
        self, epoch,
    ):
        """
        Runs the EMD and goal tests on the model.
        """
        # Log the model weights for tensorboard.
        self.writer.log_weights(model=self.model, epoch=epoch)

        # Only run tests every number of epochs.
        if epoch % SKIP_EPOCHS != 0:
            return

        # With grad off make a prediction over a random set of z-samples and the test x
        # data points.
        with torch.no_grad():
            test_size = self.x_test_pt.shape[0]
            z_test = sample_random(self.z_range, test_size, self.z_space_size)
            z_test_pt = to_tensor(z_test, self.device)
            y_pred = self.model.forward(self.x_test_pt, z_test_pt)

        # Create a numpy version of the prediction tensor.
        y_pred_d = y_pred.cpu().detach()

        # Get the z-sample predictions for every test data point.
        y_predict_mat = self.model.get_z_sample_preds(
            x=self.x_test_pt, z_samples=self.z_samples,
        )

        y_predict_mat_d = y_predict_mat.cpu().detach().numpy()

        self.plotter.initialize(epoch)
        # First test: calculate the emd.
        mean_emd = self.calculate_emd(y_pred_d)
        # Second test: Test training goal 2.
        mon_incr = self.test_goal2()
        # Second test: Test training goal 1.
        mean_goal1 = self.test_goal1(y_predict_mat, mon_incr)

        self.plotter.plot_datasets(y_pred_d, y_predict_mat_d, self.x_orderings_np)
        self.plotter.finalize(epoch)

        self.writer.log_emd(mean_emd, epoch)
        self.writer.log_goal1_error(mean_goal1, epoch)
        self.writer.log_plot(self.plotter.figures, epoch)