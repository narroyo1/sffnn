"""
This module contains class StochasticFFNN.
"""

import torch
import torch.nn as nn


DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_OUT_SIZE = 1


class StochasticFFNN(nn.Module):
    """
    This is the neural network model.
    """

    def __init__(self, z_space_size, x_space_size, device, **kwargs):
        # Perform initialization of the pytorch superclass
        super(StochasticFFNN, self).__init__()
        self.device = device
        self.hidden_size = kwargs.get("hidden_size", DEFAULT_HIDDEN_SIZE)
        self.out_size = kwargs.get("out_size", DEFAULT_OUT_SIZE)

        # Define layer types
        self.linear1 = nn.Linear(x_space_size + z_space_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x, z):
        """
        This method runs a forward pass through the model with the provided x and z.
        """

        x = torch.cat((x.view(x.size(0), -1), z.view(z.size(0), -1)), dim=1)

        x = self.linear1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.linear2(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.linear3(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)

        x = self.linear4(x)

        return x

    def get_z_sample_preds(self, x, z_samples):
        """
        This method evaluates the model over every point in x once for every sample in z_samples.
        """
        z_samples_size = z_samples.shape[0]
        # Turn off grad while we calculate our targets.
        with torch.no_grad():
            # Create a tensor with all the z samples repeated once for every element in the
            # batch. This will be matched with x_ex when finding targets.
            # [[z0[0], z0[1], ...], -> for x sample 0
            #  [z0[0], z0[1], ...], -> for x sample 1
            #  ...
            #  [z1[0], z1[1], ...], -> for x sample 0
            #  [z1[0], z1[1], ...], -> for x sample 1
            #  ...]
            z_samples_ex = torch.repeat_interleave(z_samples, x.shape[0], dim=0).to(
                device=self.device
            )

            # Create a tensor with a copy of the elements in the batch for every z sample. This
            # will be matched with z_samples_ex when finding targets.
            # [[x0], -> for z sample 0
            #  [x1], -> for z sample 0
            #  ...
            #  [x0], -> for z sample 1
            #  [x1], -> for z sample 1
            #  ...]
            x_ex = torch.cat(z_samples_size * [x]).to(device=self.device)
            # x_ex_mat = (
            #    x_ex.cpu()
            #    .detach()
            #    .numpy()
            #    .reshape(z_samples_size, x.shape[0], x.shape[1])
            # )

            # Run the model with all the elements x on every z sample.
            # [[y <- x0 z0],
            #  [y <- x1 z0],
            # ...
            #  [y <- x0 z1],
            #  [y <- x1 z1],
            # ...]
            y_predict = self.forward(x_ex, z_samples_ex)

            # Create a matrix view of the results with a column for every element and a row for
            # every z sample.
            # [[y <- x0 z0, y <- x1 z0, ...],
            #  [y <- x0 z1, y <- x1 z1, ...],
            #  ...]
            y_predict_mat = y_predict.view(
                z_samples_size, x.shape[0], y_predict.shape[1]
            )

            return y_predict_mat
