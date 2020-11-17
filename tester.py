"""
This module contains class Tester.
"""

from emd_test import EMDTest
from goal1_test import Goal1Test
from goal2_test import Goal2Test

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

        self.skip_epochs = experiment["skip_epochs"]

        if experiment.get("emd_test", True):
            self.emd_test = EMDTest(
                z_samples, datasets, plotter, self.writer, model, device
            )
        else:
            self.emd_test = None

        if experiment.get("goal1_test", True):
            self.goal1_test = Goal1Test(
                z_samples, datasets, plotter, self.writer, model, device,
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
            self.emd_test.step(epoch)
        if self.goal1_test:
            self.goal1_test.step(epoch)
        if self.goal2_test:
            self.goal2_test.step()
        self.plotter.end_frame(epoch)

        # self.writer.log_plot(self.plotter.figures, epoch)
