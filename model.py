"""
This module contains class Model.
"""
# pylint: disable=bad-continuation

import torch
import torch.nn as nn


class ZSamplePredsMixin:
    """
    This mixin adds a method to get predictions for a set of z-samples.
    """

    def get_z_sample_preds(self, x_pt, z_samples):
        """
        This method evaluates the model over every point in x once for every sample in z_samples.
        """
        z_samples_size = z_samples.shape[0]
        # Create a tensor with all the z samples repeated once for every element in the
        # batch. This will be matched with x_ex when finding targets.
        # [z0, -> for x datapoint 0
        #  z0, -> for x datapoint 1
        #  ...
        #  z0, -> for x datapoint n
        #  z1, -> for x datapoint 0
        #  z1, -> for x datapoint 1
        #  ...
        #  z1, -> for x datapoint n
        #  ...
        #  zS, -> for x datapoint 0
        #  zS, -> for x datapoint 1
        #  ...
        #  zS, -> for x datapoint n]
        # dimensions: (data points * z-samples, z-sample dimensions)
        z_samples_ex = torch.repeat_interleave(z_samples, x_pt.shape[0], dim=0).to(
            device=self.device
        )

        # Create a tensor with a copy of the elements in the batch for every z sample. This
        # will be matched with z_samples_ex when finding targets.
        # [x0, -> for z sample 0
        #  x1, -> for z sample 0
        #  ...
        #  xn, -> for z sample 0
        #  x0, -> for z sample 1
        #  x1, -> for z sample 1
        #  ...
        #  xn, -> for z sample 1
        #  ...
        #  x0, -> for z sample S
        #  x1, -> for z sample S
        #  ...
        #  xn, -> for z sample S]
        # dimensions: (data points * z-samples, input dimensions)
        x_ex = torch.cat(z_samples_size * [x_pt]).to(device=self.device)

        return self.get_preds(x_ex, z_samples_ex, x_pt.shape[0], z_samples_size)

    def get_sample_preds(self, x_pt, z_samples):
        assert x_pt.shape[0] == z_samples.shape[1]
        z_samples_size = z_samples.shape[0]

        # Create a tensor with a copy of the elements in the batch for every z sample. This
        # will be matched with z_samples_ex when finding targets.
        # [x0, -> for z sample 0
        #  x1, -> for z sample 0
        #  ...
        #  xn, -> for z sample 0
        #  x0, -> for z sample 1
        #  x1, -> for z sample 1
        #  ...
        #  xn, -> for z sample 1
        #  ...
        #  x0, -> for z sample S
        #  x1, -> for z sample S
        #  ...
        #  xn, -> for z sample S]
        # dimensions: (data points * z-samples, input dimensions)
        x_ex = torch.cat(z_samples_size * [x_pt]).to(device=self.device)

        z_samples_ex = z_samples.view(
            z_samples.shape[0] * z_samples.shape[1], z_samples.shape[2]
        )

        return self.get_preds(x_ex, z_samples_ex, x_pt.shape[0], z_samples_size)

    def get_preds(self, x_ex, z_samples_ex, input_size, z_samples_size):
        # Turn off grad while we get our predictions.
        with torch.no_grad():
            # Run the model with all the elements x on every z sample.
            # [y <- x0 z0,
            #  y <- x1 z0,
            #  ...
            #  y <- xn z0,
            #  y <- x0 z1,
            #  y <- x1 z1,
            #  ...
            #  y <- xn z1
            #  ...
            #  y <- x0 zS,
            #  y <- x1 zS,
            #  ...
            #  y <- xn zS]
            # dimensions: (data points * z-samples, output dimensions)
            y_predict = self.forward_z(x_ex, z_samples_ex)

            # Create a matrix view of the results with a column for every element and a row for
            # every z sample.
            # [[y <- x0 z0, y <- x1 z0, ..., y <- xn z0],
            #  [y <- x0 z1, y <- x1 z1, ..., y <- xn z1],
            #  ...,
            #  [y <- x0 zS, y <- x1 zS, ..., y <- xn zS]]
            # dimensions: (z-samples, data points, output dimensions)
            y_predict_mat = y_predict.view(
                z_samples_size, input_size, y_predict.shape[1]
            )

            return y_predict_mat


DEFAULT_HIDDEN_SIZE = 512
# DEFAULT_HIDDEN_SIZE = 1024
# DEFAULT_HIDDEN_SIZE = 256


class Model(nn.Module, ZSamplePredsMixin):
    """
    This is the neural network model.
    """

    def __init__(
        self, z_space_size, x_space_size, device, hidden_size=DEFAULT_HIDDEN_SIZE
    ):
        super().__init__()
        # Perform initialization of the pytorch superclass
        super(Model, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        # Define layer types
        self.linear1 = nn.Linear(x_space_size + z_space_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear31 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.linear32 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, z_space_size)

    def forward_z(self, x_pt, z_pt):
        """
        This method runs a forward pass through the model with the provided input x
        and z-samples.
        """

        x_pt = torch.cat(
            (x_pt.view(x_pt.size(0), -1), z_pt.view(z_pt.size(0), -1)), dim=1
        )

        return self.forward(x_pt)

    def forward(self, x_pt):
        """
        This method runs a forward pass through the model with the provided input x.
        """

        x_pt = self.linear1(x_pt)
        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        x_pt = self.linear2(x_pt)
        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        x_pt = self.linear3(x_pt)
        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        x_pt = self.linear31(x_pt)
        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        # x_pt = self.linear32(x_pt)
        # x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)

        x_pt = self.linear4(x_pt)

        return x_pt
