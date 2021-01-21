"""
This module contains class Trainer.
"""

import numpy as np
import torch

from torch import optim

# pylint: disable=bad-continuation, not-callable

EPSILON = np.finfo(np.float32).eps * 1000


def assert_unit_vector(unit_vector):
    last_dimension = len(unit_vector.shape) - 1
    squared = unit_vector ** 2
    summation = torch.sum(squared, dim=last_dimension)
    length = torch.sqrt(summation)

    err = torch.abs(length - 1.0)
    if (err >= EPSILON).any():
        assert False


def get_unit_vector_and_magnitude(difference):
    last_dimension = len(difference.shape) - 1
    # dimensions: (z-samples, data points, output dimensions)
    squared = difference ** 2
    # dimensions: (z-samples, data points)
    summation = torch.sum(squared, dim=last_dimension)
    # dimensions: (z-samples, data points)
    length = torch.sqrt(summation)
    # dimensions: (z-samples, data points, output dimensions)
    D = difference / length.unsqueeze(last_dimension)
    assert_unit_vector(D)

    return D, length


def dot_product_batch(X, Y):
    view_size = X.shape[0] * Y.shape[1]
    # The dot product of a unit vector with itself is 1.0.
    result = torch.bmm(
        X.view(view_size, 1, X.shape[2]), Y.view(view_size, Y.shape[2], 1)
    )
    return result.view((X.shape[0], X.shape[1]))


class MovementScalarCalculator:

    SLOT_SIZE = np.pi / 15.0

    def __init__(self, z_samples_radio, device):
        self.z_samples_radio = z_samples_radio
        self.device = device

    def get_slot_unit_vectors(self, D):
        """
        This method takes a unit vector and returns 2 unit vectors defining the slot it
        belongs to.
        """
        # Transform the unit vector into an angle. range [-pi, pi]
        angles = torch.atan2(D[:, :, 1], D[:, :, 0])

        # Transform the angle to the lower bound slot index. Dividing by the slot size and then
        # casting to integer. Add pi to keep it positive.
        slots = torch.tensor(
            (angles + np.pi) / self.SLOT_SIZE, dtype=torch.long, device=self.device
        )
        # Convert back to float.
        slots = torch.tensor(slots, dtype=torch.float32, device=self.device)

        # Convert to angles.
        # dimensions: (z-samples, data points)
        open_angles = slots * self.SLOT_SIZE - np.pi

        D1 = torch.zeros(D.shape, device=self.device)
        D2 = torch.zeros(D.shape, device=self.device)
        D3 = torch.zeros(D.shape, device=self.device)

        D1[:, :, 0] = torch.cos(open_angles)
        D1[:, :, 1] = torch.sin(open_angles)
        D2[:, :, 0] = torch.cos(open_angles + self.SLOT_SIZE)
        D2[:, :, 1] = torch.sin(open_angles + self.SLOT_SIZE)
        assert_unit_vector(D1)
        assert_unit_vector(D2)
        D3[:, :, 0] = torch.cos(open_angles + self.SLOT_SIZE / 2.0)
        D3[:, :, 1] = torch.sin(open_angles + self.SLOT_SIZE / 2.0)

        return D1, D2, D3

    def get_distances(self, D, z_samples):

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
        c = dot_product_batch(O, O) - self.z_samples_radio ** 2
        # dimensions: (z-samples, data points, output dimensions)
        discriminant = b * b - 4 * a * c
        # Every ray should intersect the circle twice.
        if (discriminant <= 0).any():
            assert False

        # dimensions: (z-samples, data points, output dimensions)
        # Elements in t_pos will always be positive and elements in t_neg will always be
        # negative.
        t_pos = (-b + torch.sqrt(discriminant)) / (2 * a)
        ########################
        if (t_pos <= -EPSILON).any():
            assert False
        ########################
        t_neg = (-b - torch.sqrt(discriminant)) / (2 * a)
        ########################
        assert (t_neg < EPSILON).all()
        ########################

        # Check that adding the distances to the origins gives a point in the circumference
        ########################
        x_pos = O + t_pos.unsqueeze(2) * D
        err = torch.abs(torch.sqrt(torch.sum(x_pos ** 2, dim=2)) - self.z_samples_radio)
        assert (err < EPSILON).all()
        x_neg = O + t_neg.unsqueeze(2) * D
        err = torch.abs(torch.sqrt(torch.sum(x_neg ** 2, dim=2)) - self.z_samples_radio)
        assert (err < EPSILON).all()
        ########################

        return t_pos, t_neg, x_pos, x_neg

    # @staticmethod
    def get_areas(self, t_1, t_2, x_1, x_2):
        triangle = t_1 * t_2 * np.sin(self.SLOT_SIZE) * 0.5

        Dx_1, _ = get_unit_vector_and_magnitude(x_1)
        Dx_2, _ = get_unit_vector_and_magnitude(x_2)
        # angles = torch.acos(dot_product_batch(Dx_1, Dx_2))
        epsilon = 1e-8
        angles = torch.acos(
            torch.clamp(dot_product_batch(Dx_1, Dx_2), -1 + epsilon, 1 - epsilon)
        )

        segment = (angles - torch.sin(angles)) * 0.5 * self.z_samples_radio ** 2

        return triangle + segment

    def calculate_scalars(self, y_pt, z_y_match, z_samples, outer_level):
        """
        Calculate the movement scalars for every difference.
        """

        # Get the unit vector of the differences.
        # D, _ = get_unit_vector_and_magnitude(difference)
        D, _ = get_unit_vector_and_magnitude(
            z_y_match.unsqueeze(0) - z_samples.unsqueeze(1)
        )

        # Get the unit vectors of the slots of each difference.
        D1, D2, D3 = self.get_slot_unit_vectors(D)

        # Get the positive and negative distances for each difference.
        # dimensions: (z-samples, data points)
        t_pos_1, t_neg_1, x_pos_1, x_neg_1 = self.get_distances(D1, z_samples)
        # dimensions: (z-samples, data points)
        t_pos_2, t_neg_2, x_pos_2, x_neg_2 = self.get_distances(D2, z_samples)

        # Get the area for the slot of each difference.
        # dimensions: (z-samples, data points)
        area_pos = self.get_areas(t_pos_1, t_pos_2, x_pos_1, x_pos_2)
        # dimensions: (z-samples, data points)
        area_neg = self.get_areas(t_neg_1, t_neg_2, x_neg_1, x_neg_2)
        # area_pos[area_pos < area_neg] *= 0.666

        # crossarea = area_pos + area_neg
        # crossarearatio = crossarea / (
        #    2.0 * 0.5 * np.sin(self.SLOT_SIZE) * z_samples_radio ** 2
        # )
        # This doesn't seem to help.
        # crossarearatio = (
        #    2.0 * 0.5 * np.sin(self.SLOT_SIZE) * self.z_samples_radio ** 2
        # ) / crossarea
        ########################
        # assert (crossarearatio < 1.0).all()
        ########################
        # A difference is in the positive outer level if it is in outer level and the positive
        # area is 0.
        outer_level0 = (torch.abs(area_pos) < EPSILON) & outer_level.unsqueeze(1)
        # A difference is in the negative outer level if it is in the outer level and the
        # positive area is greater than 0.
        outer_level1 = (area_pos > EPSILON) & outer_level.unsqueeze(1)

        # w_bp = crossarearatio / (2 * area_pos)
        w_bp = 1 / (2 * area_pos)
        w_bp[
            outer_level0
        ] = 5.0  # crossdist[outer_level0] * z_samples.outer_level_scalar
        w_bp[outer_level1] = 0.0

        # w_bp[~outer_level] = w_bp[~outer_level] * (
        #    torch.max(w_bp[~outer_level]) / torch.max(w_bp[~outer_level], dim=1).values
        # ).unsqueeze(1)
        if torch.isnan(w_bp).any():
            assert False
        assert not torch.isinf(w_bp).any()

        return w_bp, D3

    def rotate_vectors(self, vectors, angle):
        Dxx = torch.zeros(vectors.shape, device=self.device)
        Dxx[:, :, 0] = (
            np.cos(angle) * vectors[:, :, 0] - np.sin(angle) * vectors[:, :, 1]
        )
        Dxx[:, :, 1] = (
            np.sin(angle) * vectors[:, :, 0] + np.cos(angle) * vectors[:, :, 1]
        )

        return Dxx

    # @staticmethod
    def get_areas1(self, x_1, x_2, pos_greater):

        Dx_1, _ = get_unit_vector_and_magnitude(x_1)
        Dx_2, _ = get_unit_vector_and_magnitude(x_2)
        # angles = torch.acos(dot_product_batch(Dx_1, Dx_2))
        epsilon = 1e-8
        angles = torch.acos(
            torch.clamp(dot_product_batch(Dx_1, Dx_2), -1 + epsilon, 1 - epsilon)
        )

        segment = (angles - torch.sin(angles)) * 0.5 * self.z_samples_radio ** 2

        area = np.pi * self.z_samples_radio ** 2
        segment[pos_greater] = area - segment[pos_greater]

        return segment

    def calculate_scalars1(self, difference, z_samples, outer_level):
        """
        Calculate the movement scalars for every difference.
        """

        # Get the unit vector of the differences.
        D, _ = get_unit_vector_and_magnitude(difference)

        # D1 = self.rotate_vectors(D, np.pi / 2.0)
        # D2 = self.rotate_vectors(D, -np.pi / 2.0)

        # Get the positive and negative distances for each difference.
        # dimensions: (z-samples, data points)
        # t_pos_1, t_neg_1, x_pos_1, x_neg_1 = self.get_distances(D1, z_samples)
        # dimensions: (z-samples, data points)
        # t_pos_2, t_neg_2, x_pos_2, x_neg_2 = self.get_distances(D2, z_samples)
        # dimensions: (z-samples, data points)
        t_pos, t_neg, x_pos, x_neg = self.get_distances(D, z_samples)

        # Get the area for the slot of each difference.
        # dimensions: (z-samples, data points)
        # area_pos = self.get_areas1(x_pos_1, x_pos_2, t_pos > -t_neg)
        # area_pos = t_pos
        # dist = t_pos / (t_pos - t_neg)
        # area_pos = 0.333 * dist ** 3
        # area_pos = 0.333 * t_pos ** 3
        ###########################################
        # area_pos = t_pos ** 2
        ###########################################
        area_pos = np.pi * 0.5 * t_pos ** 2
        ###########################################
        # area_pos = 0.333 * t_pos ** 3
        # area_neg = -0.333 * t_neg ** 3
        # area_pos = area_pos / (area_pos + area_neg)
        ###########################################
        # dimensions: (z-samples, data points)
        # area_neg = self.get_areas1(x_neg_1, x_neg_2)
        # area_pos[area_pos < area_neg] *= 0.666

        # crossarea = area_pos + area_neg
        # crossarearatio = crossarea / (
        #    2.0 * 0.5 * np.sin(self.SLOT_SIZE) * z_samples_radio ** 2
        # )
        # This doesn't seem to help.
        # crossarearatio = (
        #    2.0 * 0.5 * np.sin(self.SLOT_SIZE) * self.z_samples_radio ** 2
        # ) / crossarea
        ########################
        # assert (crossarearatio < 1.0).all()
        ########################
        # A difference is in the positive outer level if it is in outer level and the positive
        # area is 0.
        outer_level0 = (torch.abs(area_pos) < EPSILON) & outer_level.unsqueeze(1)
        # A difference is in the negative outer level if it is in the outer level and the
        # positive area is greater than 0.
        outer_level1 = (area_pos > EPSILON) & outer_level.unsqueeze(1)

        # w_bp = crossarearatio / (2 * area_pos)
        w_bp = 1 / (2 * area_pos)
        w_bp[
            outer_level0
        ] = 0.1  # torch.mean(w_bp[~torch.isinf(w_bp)])  # 1.0  # crossdist[outer_level0] * z_samples.outer_level_scalar
        w_bp[outer_level1] = 0.0

        # w_bp[~outer_level] = w_bp[~outer_level] * (
        #    torch.max(w_bp[~outer_level]) / torch.max(w_bp[~outer_level], dim=1).values
        # ).unsqueeze(1)
        if torch.isnan(w_bp).any():
            assert False
        if torch.isinf(w_bp).any():
            assert False

        return w_bp, D


def angle_differences(vectors_to, vectors_from):
    angles = torch.atan2(vectors_to[:, 1], vectors_to[:, 0]) - torch.atan2(
        vectors_from[:, 1], vectors_from[:, 0]
    )
    angles[angles < 0.0] += 2 * np.pi
    angles[angles > np.pi] = -2 * np.pi + angles[angles > np.pi]

    return angles


class Trainer:
    """
    This class implements a mechanism to train a neural network that produces
    stochastic outputs.
    """

    def __init__(
        self, experiment, z_samples, model, device,
    ):
        self.z_samples = z_samples
        self.scalar_calculator = MovementScalarCalculator(
            z_samples.z_samples_radio, device
        )
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
            # MSE
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

    def rotate_vectors(self, vectors, angles):
        # angles[angles > np.pi] = -2 * np.pi + angles[angles > np.pi]
        # angles[angles < -np.pi] = -2 * np.pi + angles[angles < -np.pi]
        Dxx = torch.zeros(vectors.shape, device=self.device)
        Dxx[:, 0] = (
            torch.cos(angles) * vectors[:, 0] - torch.sin(angles) * vectors[:, 1]
        )
        Dxx[:, 1] = (
            torch.sin(angles) * vectors[:, 0] + torch.cos(angles) * vectors[:, 1]
        )

        return Dxx

    def calculate_z_match(self, x_pt, y_pt):
        z_pt = torch.zeros(y_pt.shape, device=self.device)
        inc = torch.ones(y_pt.shape, device=self.device)
        turn = torch.zeros(y_pt.shape, device=self.device)
        sign = None
        with torch.no_grad():
            for _ in range(113):# exit when you're good, selection without borders at first
                y_predict_mat = self.model.forward_z(x_pt, z_pt)

                differences = y_pt - y_predict_mat
                prev_sign = sign
                sign = (((differences > 0) + 0) * 2) - 1
                if prev_sign is not None:
                    bb = torch.logical_and(sign == prev_sign, turn == 0.0)
                    inc[bb] *= 2.0
                    inc = torch.clamp(inc, 0, 1000000)
                    turn[sign != prev_sign] = 1.0
                    inc[sign != prev_sign] *= 0.5
                z_pt += inc * sign
                #z_pt = torch.clamp(z_pt, -10.0, 10.0)
                total_diffs = torch.sqrt(torch.sum(differences ** 2, dim=1))
        print(torch.sum(total_diffs), torch.max(total_diffs))

        return z_pt

    def calculate_z_match1(self, x_pt, y_pt):

        z_pt = torch.zeros(y_pt.shape, device=self.device)
        z_pt1 = torch.zeros(y_pt.shape, device=self.device)
        z_pt1[:, 0] = 1.0
        inc = torch.ones(y_pt.shape[0], device=self.device)
        inc[:] = 8.0
        lengths = torch.ones(y_pt.shape[0], device=self.device)
        sign = None
        multiplier = 0.5

        with torch.no_grad():
            for _ in range(139):
                y_predict_mat = self.model.forward_z(x_pt, z_pt)
                if torch.isnan(y_predict_mat).any():
                    assert False

                differences = y_pt - y_predict_mat
                err_angles = angle_differences(y_pt, y_predict_mat)

                angles = err_angles * multiplier
                z_pt1 = self.rotate_vectors(z_pt1, angles)
                assert_unit_vector(z_pt1)
                #continue

                prev_sign = sign
                diffs1 = torch.sum(y_predict_mat ** 2, dim=1)
                diffs2 = torch.sum(y_pt ** 2, dim=1)
                diffs = diffs2 - diffs1
                sign = diffs / torch.abs(diffs)
                sign[diffs == 0] = 0
                if torch.isnan(sign).any():
                    assert False
                if prev_sign is not None:
                    # inc[sign == prev_sign] *= 2.0
                    inc[sign != prev_sign] *= 0.5
                lengths += inc * sign
                lengths = torch.clamp(lengths, 0.0, 900.0)
                z_pt = z_pt1 * lengths.unsqueeze(1)

        total_diff = torch.sum(torch.sqrt(torch.sum(differences ** 2, dim=1)))
        print(total_diff, torch.sum(torch.abs(err_angles)),
            torch.max(torch.abs(err_angles)))

        return z_pt

    def batch(self, x_pt, y_pt):#, step):
        """
        This method is called once for every training batch in an epoch.
        """
        z_y_match = self.calculate_z_match(x_pt, y_pt)

        # Get the z-samples to be used for this batch.
        z_samples, outer_level = self.z_samples.selection()#percentage)
        # Calculate the prediction matrix using the batch data and the z-samples.
        # dimensions: (z-samples, data points, output dimensions)
        y_predict_mat = self.model.get_z_sample_preds(x_pt=x_pt, z_samples=z_samples)

        w_bp, D = self.scalar_calculator.calculate_scalars(
            y_pt, z_y_match, z_samples, outer_level
        )
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
