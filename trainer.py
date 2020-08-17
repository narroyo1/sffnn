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
        self, z_samples, movement, model, learning_rate, milestones, gamma, device,
    ):
        self.z_samples = z_samples.z_samples
        self.movement = movement
        self.model = model
        self.device = device
        self.scalars = torch.tensor(
            np.stack((z_samples.less_scalar_bias, z_samples.more_scalar_bias)),
            dtype=torch.float64,
        ).to(device=self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.milestones = milestones

        # Create an adam optimizer.
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create a scheduler to decrease the learning rate.
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma,
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
        return "{}/{}/{}/{}".format(
            self.learning_rate, self.movement, self.milestones, self.gamma
        )

    def step(self, epoch):
        """
        This method is called once for every epoch.
        """
        # pylint: disable=unused-argument
        self.scheduler.step()

        # if (epoch + 1) % 10 == 0:
        #    self.movement *= 0.5

    def batch(self, x_pt, y_pt):
        """
        This method is called once for every training batch in an epoch.
        """
        # Calculate the prediction matrix using the batch data and the z-samples.
        # z-samples, batch, output
        y_predict_mat = self.model.get_z_sample_preds(x=x_pt, z_samples=self.z_samples)

        # This matrix tells if the training data is greater than the prediction.
        greater_than = torch.gt(y_pt, y_predict_mat) + 0

        ind = torch.arange(len(self.z_samples), device=self.device)#.unsqueeze(1)
        # self.scalars great/less, output, zsamples
        w_bp = self.scalars[greater_than, 0, ind]
        w_bp = w_bp.reshape((-1)).unsqueeze(1)

        y_bp = y_predict_mat + ((greater_than * 2) - 1) * self.movement
        y_bp = y_bp.reshape((-1)).unsqueeze(1)

        z_samples_bp = torch.repeat_interleave(self.z_samples, x_pt.shape[0], dim=0).to(
            device=self.device
        )

        x_bp = x_pt.repeat(*self.z_samples.shape)

        # Run backpropagation with the calculated targets.
        self.backprop(x_bp=x_bp, z_samples_bp=z_samples_bp, y_bp=y_bp, w_bp=w_bp)

    def backprop(self, x_bp, z_samples_bp, y_bp, w_bp):
        """
        This method runs backpropagation on the neural network.
        """

        # Run the forward pass.
        y_predict = self.model.forward(x_bp, z_samples_bp)

        # Compute the loss.
        loss = self.loss_fn(y_predict, y_bp, w_bp)

        # Zero out all the gradients before running the backward pass.
        self.optimizer.zero_grad()

        # Run the backward pass.
        loss.backward()

        # Run the optimizer.
        self.optimizer.step()
