"""
This module contains class Tester.
"""

from emd_test import EMDTest
from goal1_test import Goal1Test
from goal2_test import Goal2Test
from writer import Writer

# pylint: disable=bad-continuation


class Tester:
    """
    This class implements a mechanism to test the performance of a neural network
    that produces stochastic outputs.
    """

    def __init__(
        self, z_samples, datasets, trainer, plotter, model, skip_epochs, device,
    ):
        self.plotter = plotter
        self.model = model

        self.skip_epochs = skip_epochs

        self.writer = Writer(
            datasets_target_function=datasets.target_function_desc,
            trainer_params=trainer.params_desc,
            datasets_params=datasets.params_desc,
        )

        self.emd_test = EMDTest(
            z_samples, datasets, plotter, self.writer, model, device
        )
        self.goal1_test = Goal1Test(
            z_samples, datasets, plotter, self.writer, model, device,
        )
        self.goal2_test = Goal2Test(z_samples, datasets, plotter, model, device)

    def step(
        self, epoch,
    ):
        """
        Runs the tests on the model.
        """
        # Log the model weights for tensorboard.
        self.writer.log_weights(model=self.model, epoch=epoch)

        # Only run tests every number of epochs.
        if epoch % self.skip_epochs != 0:
            return

        self.plotter.start_frame(epoch)
        self.emd_test.step(epoch)
        self.goal1_test.step(epoch)
        self.goal2_test.step()
        self.plotter.end_frame(epoch)

        # self.writer.log_plot(self.plotter.figures, epoch)
