"""
This module has a set of preset experiments to test the model.
"""
import numpy as np

import functions as func

from datasets import DataSets

EXPERIMENT_1 = {
    # datasets
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": func.binder(func.fn_x2, multiplier=5.0),
    "noise_function": func.binder(func.fn_normal, std=26.5),
    # zsamples
    "outer_level_scalar": 0.2,
    "z_samples_per_dimension": np.array([13]),
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    # "num_z_samples": 13,
    # tester
    "skip_epochs": 5,
    # trainer
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "gamma": 0.5,
    "milestones": [60, 120, 180],
    # main
    "num_epochs": 181,
}

EXPERIMENT_2 = {
    # datasets
    "x_range_train": np.array([[-9.0, 5.0]]),
    "x_range_test": np.array([[-8.0, 4.0]]),
    "base_function": func.binder(func.fn_x3_x2),
    "noise_function": func.binder(func.fn_truncnormal, std=39, low=-10, upp=90),
    # zsamples
    "outer_level_scalar": 0.2,
    "z_samples_per_dimension": 13,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    # tester
    "skip_epochs": 5,
    # trainer
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "gamma": 0.5,
    "milestones": [60, 120, 180],
    # main
    "num_epochs": 181,
}

EXPERIMENT_3 = {
    # datasets
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": func.binder(func.fn_double_sin, amplitude=2.5),
    "noise_function": func.binder(func.fn_sinnormal, amplitude=2.0),
    # zsamples
    "outer_level_scalar": 0.2,
    "z_samples_per_dimension": 18,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    # tester
    "skip_epochs": 5,
    # trainer
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "gamma": 0.5,
    "milestones": [60, 120, 180, 240],
    # main
    "num_epochs": 261,
}

EXPERIMENT_4 = {
    # datasets
    "x_range_train": np.array([[-5.0, 5.0]]),
    "x_range_test": np.array([[-4.0, 4.0]]),
    "base_function": func.binder(func.fn_branch),
    "noise_function": func.binder(func.fn_normal, std=0.5),
    # zsamples
    "outer_level_scalar": 0.2,
    "z_samples_per_dimension": 18,
    "z_ranges_per_dimension": np.array([[-10.0, 10.0]]),
    # tester
    "skip_epochs": 5,
    # trainer
    "movement": 10.0,
    "learning_rate": 1e-2 / 12,
    "gamma": 0.5,
    "milestones": [60, 120, 180, 240, 300],
    # main
    "num_epochs": 321,
}

EXPERIMENT_5 = {
    # This is the range of values for input x.
    "x_range_train": np.array([[-5.0, 5.0], [-4.0, 4.0]]),
    # This is the range of values for input x on the test data set.
    "x_range_test": np.array([[-4.0, 4.0], [-3.0, 3.0]]),
    "base_function": func.binder(func.fn_x0_2_x1_2, x_space_size=2),
    "noise_function": func.binder(func.fn_halfnormal, std=5.0),
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
    "milestones": [60, 120, 180, 240, 300],
}

EXPERIMENT_6 = {
    # zsamples
    "outer_level_scalar": 0.1,
    "z_samples_per_dimension": 6,
    "z_ranges_per_dimension": np.array([[10.0, 20.0]]),
    # tester
    "skip_epochs": 10,
    # trainer
    "movement": 1.0,
    "learning_rate": 1e-2 / 4,
    "gamma": 0.85,
    "milestones": [60, 120, 180, 240, 300, 360, 420, 480],
    # main
    "num_epochs": 501,
    # datasets
    "dataset_builder": DataSets.california_housing_dataset,
}

EXPERIMENT_7 = {
    "x_range_train": np.array([[0.0, 1.0]]),
    "x_range_test": np.array([[0.0, 0.8]]),
    "base_function": func.binder(func.fn_2out_linear),
    "noise_function": func.binder(func.fn_normal2d, std=10.5),
    # "outer_level_scalar": 0.2,
    "skip_epochs": 5,
    "z_samples_per_dimension": np.array([12, 12]),
    """
    "z_samples": np.array(
        [
            [-5.0, -5.0],
            [-5.0, 0.0],
            [-5.0, 5.0],
            [0.0, -5.0],
            [0.0, 0.0],
            [0.0, 5.0],
            [5.0, -5.0],
            [5.0, 0.0],
            [5.0, 5.0],
        ]
    ),
    """: "",
    "z_samples_radio": 10.0,
    "movement": 1.0,
    "learning_rate": 1e-3,
    "num_epochs": 621,
    "gamma": 0.85,
    "milestones": [60, 120, 180, 240, 300, 360, 420, 480],
}

EXPERIMENT_DELAYS = {
    # zsamples
    "z_samples": np.array([10.2275, 11.5865, 15.0, 18.4135, 19.7725,]),
    "z_sample_labels": [
        # "$\mu - 2 * \sigma$ (2.275%)",
        # "$\mu - 1 * \sigma$ (15.865%)",
        # "$\mu$ (50%)",
        # "$\mu + 1 * \sigma$ (84.135%)",
        # "$\mu + 2 * \sigma$ (97.725%)",
        "$z_{0}$ (2.275%)",
        "$z_{1}$ (15.865%)",
        "$z_{2}$ (50%)",
        "$z_{3}$ (84.135%)",
        "$z_{4}$ (97.725%)",
        # ""
    ],
    "z_range": np.array([[10.0, 20.0]]),
    # tester
    # "skip_epochs": 50,
    "skip_epochs": 10,
    # trainer
    "movement": 1.0,
    "learning_rate": 1e-3,
    "gamma": 0.66,
    "milestones": [
        60,
        120,
        180,
        240,
        300,
        360,
        420,
        480,
        540,
        600,
        660,
        720,
        780,
        840,
        900,
        960,
        1020,
        1080,
        1140,
    ],
    # main
    "num_epochs": 1201,
    # datasets
    "dataset_builder": lambda batch_size, device: DataSets.load_csv(
        batch_size, device, "datasets/flights_filtered.csv"
    ),
    "emd_test": False,
    "goal2_test": False,
}
