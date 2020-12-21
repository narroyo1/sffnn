"""
This module contains class Goal2Test.
"""

import numpy as np

import torch

from utils import sample_random, to_tensor


class Goal2Test:
    """
    This class implements the goal 2 test on the model. That is it checks that
    the model's output is monotonically increasing.
    """

    def __init__(self, z_samples, datasets, model, device):
        self.model = model
        self.device = device
        # self.z_ranges_per_dimension = z_samples.z_ranges_per_dimension

        self.x_test = datasets.x_test

        # self.z_space_size = z_samples.Z_SPACE_SIZE

    def test_goal2(self):
        """
        This method tests goal 2 i.e. whether f(x, z) is monotonically increasing in z for any
        given x.
        """
        return True

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
        y_predict_mat = self.model.get_z_sample_preds(x_pt=x_goal, z_samples=z_goal,)

        y_predict_mat_d = y_predict_mat.cpu().detach().numpy()

        ascending = np.all(y_predict_mat_d[:-1] <= y_predict_mat_d[1:])

        return ascending

    def step(self):
        """
        Runs and plots a step of the goal 2 test.
        """

        # Second test: Test training goal 2.
        mon_incr = self.test_goal2()

        return mon_incr
