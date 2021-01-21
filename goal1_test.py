"""
This module contains class Goal1Test.
"""
# pylint: disable=no-member

import numpy as np

import torch

from utils import to_tensor
from trainer import MovementScalarCalculator

# The number of points in X to test for goal 1.
GOAL1_TEST_POINTS = 50

# This constant specifies the number of samples to be used on each sample partition.
GOAL1_SAMPLES_PER_TEST_POINT = 800


def get_angles(unit_vectors):
    """
    This function gets the angles of the give unit vectors. The range of the result
    is [-pi, pi].
    Note: unit_vectors shall be a tensor of unit vectors.
    """
    epsilon = 1e-8
    # Clamp the input to avoid nans.
    angles = torch.acos(torch.clamp(unit_vectors[:, :, 0], -1 + epsilon, 1 - epsilon))
    # Add sign information.
    angles[unit_vectors[:, :, 1] < 0.0] = -angles[unit_vectors[:, :, 1] < 0.0]
    # Add sign information and use [0, 2 * pi] range.
    # angles[unit_vectors[:, :, 1] < 0.0] = 2 * np.pi - angles[unit_vectors[:, :, 1] < 0.0]

    return angles


def angle_differences(vectors_to, vectors_from):
    angles = torch.atan2(vectors_to[:, :, 1], vectors_to[:, :, 0]) - torch.atan2(
        vectors_from[:, :, 1], vectors_from[:, :, 0]
    )
    angles[angles < 0.0] += 2 * np.pi
    angles[angles > np.pi] = -2 * np.pi + angles[angles > np.pi]

    return angles


class Goal1Test:
    """
    This class implements the goal 1 test on the model. That is it checks that
    the predictions for the z-samples have the right distribution ratios.
    """

    def __init__(self, z_samples, datasets, model, device):

        self.z_samples = z_samples
        self.model = model
        self.device = device
        self.scalar_calculator = MovementScalarCalculator(
            z_samples.z_samples_radio, device
        )

        self.y_test = datasets.y_test
        self.x_test = datasets.x_test

        self.y_test_pt = to_tensor(datasets.y_test, device)
        self.x_test_pt = to_tensor(datasets.x_test, device)
        self.x_orderings_pt = [
            torch.sort(self.x_test_pt[:, i])[1] for i in range(self.x_test_pt.shape[1])
        ]

    def compute_projections(self, samples, current_uv, nominals, radios, predict_mat):
        from trainer import get_unit_vector_and_magnitude, assert_unit_vector

        assert_unit_vector(current_uv)

        targets = samples.unsqueeze(1) + current_uv * radios.view(-1, 1, 1)

        y_radio_mat = self.model.get_sample_preds(
            x_pt=self.x_test_pt, z_samples=targets,
        )

        differences = y_radio_mat - predict_mat
        # differences_uv, _ = get_unit_vector_and_magnitude(differences)

        # ranges: [-pi, pi]
        # nominals_angles = get_angles(nominals)  # angle_differences
        # ranges: [-pi, pi]
        # differences_angles = get_angles(differences_uv)
        # ranges: [-2 * pi, 2 * pi]
        err_angles = angle_differences(nominals, differences)
        print(
            "sum err_angles",
            torch.sum(torch.abs(err_angles)),
            torch.max(torch.abs(err_angles)),
        )

        return y_radio_mat, err_angles

    def rotate_vectors(self, vectors, angles):
        # angles[angles > np.pi] = -2 * np.pi + angles[angles > np.pi]
        # angles[angles < -np.pi] = -2 * np.pi + angles[angles < -np.pi]
        Dxx = torch.zeros(vectors.shape, device=self.device)
        Dxx[:, :, 0] = (
            torch.cos(angles) * vectors[:, :, 0] - torch.sin(angles) * vectors[:, :, 1]
        )
        Dxx[:, :, 1] = (
            torch.sin(angles) * vectors[:, :, 0] + torch.cos(angles) * vectors[:, :, 1]
        )

        return Dxx

    def approximate_projected_radios(
        self, filtered_predict_mat, filtered_samples, filtered_radios
    ):
        from trainer import get_unit_vector_and_magnitude

        differences = self.y_test_pt - filtered_predict_mat
        differences_uv, _ = get_unit_vector_and_magnitude(differences)

        y_radio_mat, err_angles = self.compute_projections(
            filtered_samples,
            differences_uv,
            differences,
            filtered_radios,
            filtered_predict_mat,
        )

        current = differences_uv
        multiplier = 0.5
        for i in range(510):
            angles = err_angles * multiplier
            current = self.rotate_vectors(current, angles)
            if (i + 1) % 80 == 0:
                print("multiplier", multiplier)
                multiplier *= 0.5

            y_radio_mat, err_angles = self.compute_projections(
                filtered_samples,
                current,
                differences,
                filtered_radios,
                filtered_predict_mat,
            )
            if torch.max(torch.abs(err_angles)) < 0.00001:
                break

        return y_radio_mat

    def test_goal1(self, y_predict_mat):
        """
        This method tests the hypothesis that every z-line divides the level by half.
        """
        from trainer import get_unit_vector_and_magnitude

        radios = self.z_samples.radios
        radios_filter = radios >= self.z_samples.z_sample_spacing / 2.0
        filtered_radios = radios[radios_filter]
        filtered_predict_mat = y_predict_mat[radios_filter]
        filtered_samples = self.z_samples.samples[radios_filter]

        y_radio_mat = self.approximate_projected_radios(
            filtered_predict_mat, filtered_samples, filtered_radios
        )

        difference = y_radio_mat - filtered_predict_mat
        distances1 = torch.sum(difference ** 2, dim=2)

        difference = self.y_test_pt - filtered_predict_mat
        distances2 = torch.sum(difference ** 2, dim=2)

        less_than = (distances2 < distances1) + 0

        sums = torch.sum(less_than, dim=1)
        areas = (filtered_radios ** 2) * np.pi

        total_area = (self.z_samples.z_samples_radio ** 2) * np.pi
        # area_ratio = torch.sum(areas) / total_area
        area_ratios = areas / total_area

        # ratios = sums / y_predict_mat.shape[1]
        # ratios = sums / torch.sum(sums)
        ratios = sums / (filtered_predict_mat.shape[1] * area_ratios)

        # area = (self.z_samples.z_samples_radio ** 2) * np.pi
        # expected_ratios = areas / area
        # expected_ratios = areas / torch.sum(areas)

        from trainer import get_unit_vector_and_magnitude

        """
        w_bp, D = self.scalar_calculator.calculate_scalars(
            self.y_test_pt - y_predict_mat,
            self.z_samples.samples,
            self.z_samples.outer_level,
        )

        movement = D * w_bp.unsqueeze(2)
        distances = torch.sqrt(torch.sum(movement ** 2, dim=2))
        tm = torch.sum(distances ** 2, dim=1)
        print(
            "total_movement",
            torch.mean(tm, dim=0),
            torch.std(tm, dim=0),
            torch.max(tm, dim=0),
        )
        """
        """
        # total_movement = D * w_bp.unsqueeze(2)
        _, tm = get_unit_vector_and_magnitude(total_movement)

        # for j in range(0, total_movement.shape[1], 500):
        #    ts = torch.sum(total_movement[:, j : j + 500], dim=1)
        #    print(j)
        #    for i in range(total_movement.shape[0]):
        #        print(i, ts[i, 0], ts[i, 1])
        """

        # This is the number of test points separting group middle points.
        test_point_spacing = int(self.y_test.shape[0] / GOAL1_TEST_POINTS)

        multiplier = filtered_radios / torch.max(filtered_radios)
        multiplier = multiplier / torch.sum(multiplier)

        # This is the mean ratio of smaller than over all test data points for each z-sample ring.
        # dimensions: (z-samples)
        # goal1_mean = torch.mean(smaller_than, dim=1,)
        # This is the error for every mean ratio above.
        # dimensions: (z-samples)
        # goal1_err = goal1_mean - self.less_than_ratios
        # goal1_err = (ratios - expected_ratios) / expected_ratios
        goal1_err = 1.0 - ratios  # - expected_ratios
        # This is the absolute error for every z-sample. It has to be absolute so that
        # they doesn't cancel each other when averaging.
        # dimensions: (z-samples)
        # goal1_err_abs = torch.abs(goal1_err)
        goal1_err_abs = torch.abs(goal1_err)
        # This is the single mean value of the absolute error of all z-samples.
        # goal1_mean_err_abs = torch.mean(goal1_err_abs)
        goal1_mean_err_abs = torch.sum(goal1_err_abs * multiplier)
        print("goal1_mean_err_abs", goal1_mean_err_abs)
        # return None, None, None

        # The local errors for every dimension will be returned in this variable.
        local_goal1_errs = []
        ########################################################################
        """
        import random
        num = random.randint(0, y_radio_mat.shape[0])
        num = torch.argmax(ratios)
        print("num", num)
        """
        angle = np.linspace(-np.pi, np.pi, 5)
        angle = angle[:-1]
        # angle = np.linspace(-np.pi, np.pi, 720)
        # a = (self.z_samples.z_sample_spacing / 2.0 * np.cos(angle)).flatten()
        # b = (self.z_samples.z_sample_spacing / 2.0 * np.sin(angle)).flatten()
        a = (0.5 * np.cos(angle)).flatten()
        b = (0.5 * np.sin(angle)).flatten()

        xx = np.column_stack((a, b))
        f = filtered_samples.repeat((xx.shape[0], 1))
        f = f.cpu().detach().numpy()
        targets = f + xx.repeat(filtered_samples.shape[0], axis=0)
        xxx = self.model.get_sample_preds(
            x_pt=torch.zeros((targets.shape[0], 1), device=self.device),
            z_samples=torch.tensor(
                targets, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
        )
        # xxx = self.model.get_sample_preds(
        #    x_pt=torch.zeros((targets.shape[0], 1), device=self.device), z_samples=torch.tensor(targets, device=self.device, dtype=torch.float32).unsqueeze(0),
        # )
        ########################################################################
        return (
            goal1_mean_err_abs,
            local_goal1_errs,
            None,  # D,  # torch.mean(D, dim=1),
            None,  # w_bp,  # torch.mean(w_bp, dim=1),
            None,  # y_radio_mat,  # [num],
            xxx,  # self.y_test_pt[less_than[num] == 1],
            ratios,
            radios_filter,
        )

        num_dimensions = self.x_test.shape[1]
        for dimension in range(num_dimensions):
            # This array will have the goal error means for every test point's vicinity.
            # Add 2 members for the plot edges.
            local_goal1_err = np.zeros(GOAL1_TEST_POINTS + 2)
            local_goal1_max_err = np.zeros(GOAL1_TEST_POINTS + 2)
            # dimensions: (z-samples, test datapoint groups)
            local_goal1_err_zsample = np.zeros(
                (self.z_samples.ring_numbers, GOAL1_TEST_POINTS + 2)
            )

            x_np = np.zeros(GOAL1_TEST_POINTS + 2)
            for point in range(GOAL1_TEST_POINTS):
                # For the current test point, select the start and stop indexes.
                start = point * test_point_spacing - GOAL1_SAMPLES_PER_TEST_POINT / 2
                stop = start + GOAL1_SAMPLES_PER_TEST_POINT
                start = int(max(start, 0))
                stop = int(stop)

                ## This is the mean ratio of smaller than over all the data points in this
                ## vicinity.
                ## dimension: (z-samples)
                # smaller_than_mean = torch.mean(
                #    smaller_than[:, self.x_orderings_pt[dimension]][:, start:stop],
                #    dim=1,
                # )

                # Get the error by substracting it from the expected ratio and calculate
                # the absolute value.
                # dimension: (z-samples)
                # smaller_than_mean_abs = torch.abs(
                #    smaller_than_mean - self.less_than_ratios
                # )
                local_total_movement = torch.sum(
                    D[:, self.x_orderings_pt[dimension][start:stop]]
                    * w_bp[:, self.x_orderings_pt[dimension][start:stop]].unsqueeze(2),
                    dim=1,
                )
                # dimensions: (z-samples)
                _, local_tm = get_unit_vector_and_magnitude(local_total_movement)

                local_goal1_err[point + 1] = torch.mean(local_tm, dim=0)
                # local_goal1_err_zsample[:, point + 1] = local_tm
                local_goal1_max_err[point + 1] = 0  # torch.max(smaller_than_mean_abs)
                # Calculate the x value for the plot as the average of all data points considered.
                x_np[point + 1] = torch.mean(
                    self.x_test_pt[self.x_orderings_pt[dimension]][start:stop][
                        :, dimension
                    ]
                )

            x_np[0] = self.x_test[self.x_orderings_pt[dimension][0]][dimension]
            local_goal1_err[0] = local_goal1_err[1]
            local_goal1_max_err[0] = local_goal1_max_err[1]
            x_np[-1] = self.x_test[self.x_orderings_pt[dimension][-1]][dimension]
            local_goal1_err[-1] = local_goal1_err[-2]
            local_goal1_max_err[-1] = local_goal1_max_err[-2]

            local_goal1_errs.append((x_np, local_goal1_err, local_goal1_err_zsample))

        return goal1_mean_err_abs, local_goal1_errs

    def step(self, y_predict_mat):
        """
        Runs and plots a step of the goal 1 test.
        """

        # Second test: Test training goal 1.
        global_goal1_err, local_goal1_errs, d, l, r, p, rs, fil = self.test_goal1(
            y_predict_mat
        )

        return (
            global_goal1_err,
            local_goal1_errs,
            d.cpu().detach().numpy() if d is not None else None,
            l.cpu().detach().numpy() if l is not None else None,
            r.cpu().detach().numpy() if r is not None else None,
            p.cpu().detach().numpy() if p is not None else None,
            rs.cpu().detach().numpy() if rs is not None else None,
            fil.cpu().detach().numpy() if rs is not None else None,
        )
