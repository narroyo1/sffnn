"""
This module contains class Trainer.
"""

import numpy as np
import torch

from torch import optim

# pylint: disable=bad-continuation, not-callable

EPSILON = np.finfo(np.float32).eps * 100


def assert_unit_vector(unit_vector):
    last_dimension = len(unit_vector.shape) - 1
    squared = unit_vector ** 2
    summation = torch.sum(squared, dim=last_dimension)
    length = torch.sqrt(summation)

    err = torch.abs(length - 1.0)
    assert (err < EPSILON).all()


def get_unit_and_mag(difference):
    last_dimension = len(difference.shape) - 1
    # dimensions: (z-samples, data points, output dimensions)
    squared = difference * difference
    # dimensions: (z-samples, data points)
    summation = torch.sum(squared, dim=last_dimension)
    # dimensions: (z-samples, data points)
    length = torch.sqrt(summation)
    # dimensions: (z-samples, data points, output dimensions)
    D = difference / length.unsqueeze(last_dimension)
    assert_unit_vector(D)

    return D, length


SLOT_SIZE = np.pi / 15.0


def get_direction_slots(D, device):
    angles = torch.atan2(D[:, :, 1], D[:, :, 0])

    slots = torch.tensor((angles + np.pi) / SLOT_SIZE, dtype=torch.long, device=device)
    slots = torch.tensor(slots, dtype=torch.float32, device=device)

    open_angles = slots * SLOT_SIZE - np.pi

    D1 = torch.zeros(D.shape, device=device)
    D2 = torch.zeros(D.shape, device=device)

    D1[:, :, 0] = torch.cos(open_angles)
    D1[:, :, 1] = torch.sin(open_angles)
    D2[:, :, 0] = torch.cos(open_angles + SLOT_SIZE)
    D2[:, :, 1] = torch.sin(open_angles + SLOT_SIZE)
    assert_unit_vector(D1)
    assert_unit_vector(D2)

    return D1, D2


def get_distances(D, z_samples, z_samples_radio):
    def dot_product_batch(X, Y):
        view_size = X.shape[0] * Y.shape[1]
        # The dot product of a unit vector with itself is 1.0.
        result = torch.bmm(
            X.view(view_size, 1, X.shape[2]), Y.view(view_size, Y.shape[2], 1)
        )
        return result.view((X.shape[0], X.shape[1]))

    a = dot_product_batch(D, D)
    ########################
    err = torch.abs(a - 1.0)
    assert (err < EPSILON).all()
    ########################
    # dimensions: (z-samples, data points, output dimensions)
    O = torch.stack([z_samples] * D.shape[1], dim=1)
    # dimensions: (z-samples, data points, output dimensions)
    b = 2 * dot_product_batch(O, D)
    # dimensions: (z-samples, data points, output dimensions)
    c = dot_product_batch(O, O) - z_samples_radio ** 2
    # dimensions: (z-samples, data points, output dimensions)
    discriminant = b * b - 4 * a * c
    # Every ray should intersect the circle twice.
    assert (discriminant > 0).all()

    # dimensions: (z-samples, data points, output dimensions)
    # Elements in x0 will always be positive and elements in x1 will always be
    # negative.
    x0 = (-b + torch.sqrt(discriminant)) / (2 * a)
    assert (x0 > 0.0).all()
    x1 = (-b - torch.sqrt(discriminant)) / (2 * a)
    assert (x1 < 0.0).all()

    # Check that adding the distances to the origins gives a point in the circumference
    ########################
    ax = O + x0.unsqueeze(2) * D
    err = torch.abs(torch.sqrt(torch.sum(ax * ax, dim=2)) - z_samples_radio)
    assert (err < EPSILON).all()
    bx = O + x1.unsqueeze(2) * D
    err = torch.abs(torch.sqrt(torch.sum(bx * bx, dim=2)) - z_samples_radio)
    assert (err < EPSILON).all()
    ########################

    return x0, x1


def get_movement_scalars(D1, D2, z_samples, z_samples_radio):
    x0_1, x1_1 = get_distances(D1, z_samples, z_samples_radio)
    x0_2, x1_2 = get_distances(D2, z_samples, z_samples_radio)

    x0 = x0_1 * x0_2 * np.sin(SLOT_SIZE) * 0.5
    x1 = x1_1 * x1_2 * np.sin(SLOT_SIZE) * 0.5

    crossarea = x0 + x1
    # crossarearatio = crossarea / (
    #    2.0 * 0.5 * np.sin(SLOT_SIZE) * z_samples_radio ** 2
    # )
    # This doesn't seem to help.
    crossarearatio = (2.0 * 0.5 * np.sin(SLOT_SIZE) * z_samples_radio ** 2) / crossarea
    ########################
    # assert (crossarearatio < 1.0).all()
    ########################
    # outer_level0 = x0 <= 0.01
    # outer_level1 = x1 >= -0.01

    # x0[x0 <= 0.01] = 0.01
    #
    # w_bp = crossarearatio / (2 * x0)
    w_bp = 1 / (2 * x0)
    # w_bp[outer_level0] = crossdist[outer_level0] * z_samples.outer_level_scalar
    # w_bp[outer_level1] = 0.0
    assert not torch.isnan(w_bp).any()
    assert not torch.isinf(w_bp).any()

    return w_bp


class Trainer:
    """
    This class implements a mechanism to train a neural network that produces
    stochastic outputs.
    """

    def __init__(
        self, experiment, z_samples, model, device,
    ):
        self.z_samples = z_samples
        self.movement = experiment["movement"]
        self.model = model
        self.device = device

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
            ## MAE
            # sqerr = torch.abs(inputs - targets)
            out = sqerr * weights
            loss = out.mean()

            return loss

        # self.loss_fn = torch.nn.MSELoss()
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
        z_samples = self.z_samples.selection()
        # Calculate the prediction matrix using the batch data and the z-samples.
        # dimensions: (z-samples, data points, output dimensions)
        y_predict_mat = self.model.get_z_sample_preds(x_pt=x_pt, z_samples=z_samples)

        D, _ = get_unit_and_mag(y_pt - y_predict_mat)

        D1, D2 = get_direction_slots(D, self.device)

        w_bp = get_movement_scalars(D1, D2, z_samples, self.z_samples.z_samples_radio)
        w_bp = w_bp.view((w_bp.shape[0] * w_bp.shape[1], 1))

        # dimensions: (z-samples, data points, output dimensions)
        # y_bp = y_predict_mat + D * self.movement * crossdistratio.unsqueeze(2)
        # y_bp = y_predict_mat + D * self.movement  * w_bp.unsqueeze(2)
        y_bp = y_predict_mat + D * self.movement
        # dimensions: (z-samples * data points, output dimensions)
        y_bp = y_bp.reshape((y_bp.shape[0] * y_bp.shape[1], y_bp.shape[2]))

        # dimensions: (z-samples * data points, z-sample dimensions)
        z_samples_bp = torch.repeat_interleave(z_samples, x_pt.shape[0], dim=0).to(
            device=self.device
        )

        # dimensions: (z-samples * data points, input dimensions)
        repeat_shape = [1 for _ in range(len(x_pt.shape))]
        repeat_shape[0] = z_samples.shape[0]
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
        # loss = self.loss_fn(y_predict, y_bp)
        loss = self.loss_fn(y_predict, y_bp, w_bp)

        # Zero out all the gradients before running the backward pass.
        self.optimizer.zero_grad()

        # Run the backward pass.
        loss.backward()

        # Run the optimizer.
        self.optimizer.step()
