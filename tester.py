"""
This module contains class Tester.
"""

import numpy as np
import torch

from emd_test import EMDTest
from goal1_test import Goal1Test
from goal2_test import Goal2Test
from utils import to_tensor, sample_random

# pylint: disable=bad-continuation


class Tester:
    """
    This class implements a mechanism to test the performance of a neural network
    that produces stochastic outputs.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self, experiment, z_samples, datasets, plotter, writer, model, device,
    ):
        self.plotter = plotter
        self.writer = writer
        self.model = model
        self.device = device

        self.z_samples = z_samples.z_samples
        # self.z_range = z_samples.z_range
        # self.z_space_size = z_samples.Z_SPACE_SIZE
        self.x_test_pt = to_tensor(datasets.x_test, device)
        self.x_orderings_np = [
            np.argsort(datasets.x_test[:, i]) for i in range(datasets.x_test.shape[1])
        ]

        self.skip_epochs = experiment["skip_epochs"]

        if experiment.get("emd_test", True):
            self.emd_test = EMDTest(datasets, plotter, self.writer, device)
        else:
            self.emd_test = None

        if experiment.get("goal1_test", True):
            self.goal1_test = Goal1Test(
                z_samples, datasets, plotter, self.writer, device,
            )
        else:
            self.goal1_test = None

        if experiment.get("goal2_test", True):
            self.goal2_test = Goal2Test(z_samples, datasets, plotter, model, device)
        else:
            self.goal2_test = None

    def step(
        self, epoch,
    ):
        """
        Runs the tests on the model.
        """
        # Log the model weights for tensorboard.
        self.writer.log_weights(model=self.model, epoch=epoch)

        # Only run tests every number of epochs.
        if epoch % self.skip_epochs != 0 or epoch < 10:
            # if epoch != 267:
            return

        self.plotter.start_frame(epoch)
        if self.emd_test:
            # With grad off make a prediction over a random set of z-samples and the test x
            # data points.
            with torch.no_grad():
                test_size = self.x_test_pt.shape[0]
                z_test = sample_random(self.z_range, test_size, self.z_space_size)
                z_test_pt = to_tensor(z_test, self.device)
                y_pred = self.model.forward_z(self.x_test_pt, z_test_pt)

            # Create a numpy version of the prediction tensor.
            y_pred_d = y_pred.cpu().detach()
            self.emd_test.step(epoch, y_pred_d)

            self.plotter.plot_datasets_preds(y_pred_d)

        if self.goal1_test:
            # Get the z-sample predictions for every test data point.
            y_predict_mat = self.model.get_z_sample_preds(
                x_pt=self.x_test_pt, z_samples=self.z_samples,
            )
            self.goal1_test.step(epoch, y_predict_mat)

            y_predict_mat_d = y_predict_mat.cpu().detach().numpy()
            self.plotter.plot_datasets_zlines(y_predict_mat_d, self.x_orderings_np)

        if self.goal2_test:
            self.goal2_test.step()
        self.plotter.end_frame(epoch)

        # self.writer.log_plot(self.plotter.figures, epoch)
