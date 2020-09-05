"""
This module has a set of preset experiments to test the model.
"""
import numpy as np

from datasets import DataSets

# pylint: disable=unused-import
from functions import (
    binder,
    fn_x2,
    fn_x3_x2,
    fn_double_sin,
    fn_branch,
    fn_x0_2_x1_2,
    fn_normal,
    fn_truncnormal,
    fn_sinnormal,
    fn_halfnormal,
    fn_2out_linear,
)

EXPERIMENT_1 = {
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": binder(fn_x2, multiplier=5.0),
    "noise_function": binder(fn_normal, std=26.5),
    "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": np.array([13]),
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "num_epochs": 181,
    "gamma": 0.5,
}

EXPERIMENT_2 = {
    "x_range_train": np.array([[-9.0, 5.0]]),
    "x_range_test": np.array([[-8.0, 4.0]]),
    "base_function": binder(fn_x3_x2),
    "noise_function": binder(fn_truncnormal, std=39, low=-10, upp=90),
    "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": 13,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "num_epochs": 181,
    "gamma": 0.5,
}

EXPERIMENT_3 = {
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": binder(fn_double_sin, amplitude=2.5),
    "noise_function": binder(fn_sinnormal, amplitude=2.0),
    "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": 18,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "num_epochs": 261,
    "gamma": 0.5,
}

EXPERIMENT_4 = {
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": binder(fn_branch),
    "noise_function": binder(fn_normal, std=0.5),
    "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": 18,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "num_epochs": 321,
    "gamma": 0.5,
}

EXPERIMENT_5 = {
    # This is the range of values for input x.
    "x_range_train": np.array([[-5.0, 5.0], [-4.0, 4.0]]),
    # This is the range of values for input x on the test data set.
    "x_range_test": np.array([[-4.0, 4.0], [-3.0, 3.0]]),
    "base_function": binder(fn_x0_2_x1_2, x_space_size=2),
    "noise_function": binder(fn_halfnormal, std=5.0),
    # If this value is 1.0 it may pull the outer z-lines too far and affect
    # the z-lines immediately next to them.
    "outer_level_scalar": 0.2,
    # The number of epochs skipped before running the tests.
    "skip_epochs": 5,
    "z_samples_per_dimension": 13,
    # The range of values in z-space.
    # Having a range spanning from negative to positive can have unintended
    # training results.
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    # The total number of epochs to run training for.
    "num_epochs": 321,
    "gamma": 0.5,
}

EXPERIMENT_6 = {
    "outer_level_scalar": 0.1,
    "skip_epochs": 10,
    "z_samples_per_dimension": 6,
    "z_ranges_per_dimension": np.array([[10.0, 20.0]]),
    "movement": 1.0,
    "learning_rate": 1e-2 / 4,
    "num_epochs": 501,
    "gamma": 0.85,
    "dataset_builder": DataSets.california_housing_dataset,
}

EXPERIMENT_7 = {
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": binder(fn_2out_linear),
    "noise_function": binder(fn_normal, std=0.5),
    "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": np.array([3, 4]),
    "z_ranges_per_dimension": np.array([[-10.0, 10.0], [-5.0, 5.0]]),
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "num_epochs": 321,
    "gamma": 0.5,
}
