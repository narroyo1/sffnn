"""
This module contains class Trainer.
"""

import numpy as np
import torch

from torch import optim

# pylint: disable=bad-continuation, not-callable

# %%
class Trainer:
    """
    This class implements a mechanism to train a neural network that produces
    stochastic outputs.
    """

    def __init__(
        self, experiment, z_samples, model, device,
    ):
        self.z_samples = z_samples.z_samples
        self.movement = experiment["movement"]
        self.model = model
        self.device = device
        # dimensions: (greater/smaller, z-samples)
        self.scalars = torch.tensor(
            np.stack((z_samples.less_scalar, z_samples.more_scalar)),
            dtype=torch.float64,
        ).to(device=self.device)

        self.learning_rate = experiment["learning_rate"]
        self.gamma = experiment["gamma"]
        self.milestones = experiment["milestones"]

        # Create an adam optimizer.
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Create a scheduler to decrease the learning rate.
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.gamma,
        )

        # Create an Weighted Mean Squared Error (WMSE) loss function for the targets.
        def weighted_mse_loss(inputs, targets, weights):
            sqerr = (inputs - targets) ** 2
            out = sqerr * weights
            loss = out.mean()

            return loss

        self.loss_fn = weighted_mse_loss

    @property
    def params_desc(self):
        """ This property returns a description of the Trainer parameters. """
        return "{}/{}/{}/{}".format(
            self.learning_rate, self.movement, self.milestones, self.gamma
        )

    def step(self, epoch):
        """
        This method is called once for every epoch.
        """
        # pylint: disable=unused-argument
        self.scheduler.step()

    def batch(self, x_pt, y_pt):
        """
        This method is called once for every training batch in an epoch.
        """
        # Calculate the prediction matrix using the batch data and the z-samples.
        # dimensions: (z-samples, data points, output dimensions)
        y_predict_mat = self.model.get_z_sample_preds(
            x_pt=x_pt, z_samples=self.z_samples
        )

        # This matrix tells if the training data is greater than the prediction.
        # dimensions: (z-samples, data points, output dimensions)
        greater_than = torch.gt(y_pt.squeeze(), y_predict_mat) + 0

        ind = torch.arange(len(self.z_samples), device=self.device).unsqueeze(1)
        # This matrix will have he weights to be used on the loss function.
        # dimensions: (z-samples, data points, output dimensions)
        w_bp = self.scalars[greater_than, ind]
        # dimensions: (z-samples * data points, output dimensions)
        w_bp = w_bp.reshape((-1)).unsqueeze(1)

        # dimensions: (z-samples, data points, output dimensions)
        y_bp = y_predict_mat + ((greater_than * 2) - 1) * self.movement
        # dimensions: (z-samples * data points, output dimensions)
        y_bp = y_bp.reshape((-1)).unsqueeze(1)

        # dimensions: (z-samples * data points, z-sample dimensions)
        z_samples_bp = torch.repeat_interleave(self.z_samples, x_pt.shape[0], dim=0).to(
            device=self.device
        )

        # dimensions: (z-samples * data points, input dimensions)
        repeat_shape = [1 for _ in range(len(x_pt.shape))]
        repeat_shape[0] = self.z_samples.shape[0]
        x_bp = x_pt.repeat(tuple(repeat_shape))

        # Run backpropagation with the calculated targets.
        self.backprop(x_bp=x_bp, z_samples_bp=z_samples_bp, y_bp=y_bp, w_bp=w_bp)

    def backprop(self, x_bp, z_samples_bp, y_bp, w_bp):
        """
        This method runs backpropagation on the neural network.
        """

        # Run the forward pass.
        y_predict = self.model.forward_z(x_bp, z_samples_bp)

        # Compute the loss.
        loss = self.loss_fn(y_predict, y_bp, w_bp)

        # Zero out all the gradients before running the backward pass.
        self.optimizer.zero_grad()

        # Run the backward pass.
        loss.backward()

        # Run the optimizer.
        self.optimizer.step()
