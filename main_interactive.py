"""
This module can be used to compose a stochastic function and train a neural network
to aproximate it.
"""
# pylint: disable=invalid-name
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import time

import numpy as np

import torch


from model import StochasticFFNN
from trainer import Trainer
from tester import Tester
from zsamples import ZSamples
from datasets import DataSets
from plotter import Plotter

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
)


# %%

# Use coprime numbers to prevent any matching points between train and test.
TRAIN_SIZE = 31013
# TRAIN_SIZE = 13011
# TRAIN_SIZE = 9010
# TRAIN_SIZE = 9611

TEST_SIZE = 5007
# TEST_SIZE = 1001

# This is the range of values for input x.
# X_RANGE_TRAIN = np.array([[-5.0, 5.0], [-4.0, 4.0]])  # exp 5
X_RANGE_TRAIN = np.array([[-5.0, 5.0]])  # exp 1, 3, 4
# X_RANGE_TRAIN = np.array([[-9.0, 5.0]])  # exp 2

# This is the range of values for input x on the test data set.
# X_RANGE_TEST = np.array([[-4.0, 4.0], [-3.0, 3.0]])  # exp 5
X_RANGE_TEST = np.array([[-4.0, 4.0]])  # exp 1, 3, 4
# X_RANGE_TEST = np.array([[-8.0, 4.0]])  # exp 2


BATCH_SIZE = 2048

device = torch.device("cuda")

datasets = DataSets.generated_dataset(
    base_function=binder(fn_x2, multiplier=5.0),  # exp 1
    # base_function=binder(fn_x3_x2),  # exp 2
    # base_function=binder(fn_double_sin, amplitude=2.5), # exp 3
    # base_function=binder(fn_branch), # exp 4
    # base_function=binder(fn_x0_2_x1_2, x_space_size=2),  # exp 5
    noise_function=binder(fn_normal, std=26.5),  # exp 1
    # noise_function=binder(fn_truncnormal, std=39, low=-10, upp=90),  # exp 2
    # noise_function=binder(fn_sinnormal, amplitude=2.0), # exp 3
    # noise_function=binder(fn_normal, std=0.5), # exp 4
    # noise_function=binder(fn_halfnormal, std=5.0),  # exp 5
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    x_range_train=X_RANGE_TRAIN,
    x_range_test=X_RANGE_TEST,
    batch_size=BATCH_SIZE,
    device=device,
)
"""
datasets = DataSets.california_housing_dataset(BATCH_SIZE, device)  # exp 6
"""
# datasets.show()


NUM_Z_SAMPLES = 13  # exp 1, 2, 5
# NUM_Z_SAMPLES = 18  # exp 3, 4
# NUM_Z_SAMPLES = 6  # exp 6
# The range of values in z-space.
# Having a range spanning from negative to positive can have unintended
# training results.
Z_RANGE = np.array([[-10.0, 10.0]])  # exp 1, 2, 3, 4, 5
# Z_RANGE = np.array([[10.0, 20.0]])  # exp 6

z_samples = ZSamples(num_z_samples=NUM_Z_SAMPLES, z_range=Z_RANGE, device=device)


stochastic_ffnn = StochasticFFNN(
    z_samples.Z_SPACE_SIZE,
    len(datasets.x_dimensions),
    device=device,
    # hidden_size=1536,  # exp 6
).to(device=device)


# %%


# LEARNING_RATE = 1e-3
# LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-2 / 12  # exp 1, 2, 3, 4, 5
# LEARNING_RATE = 1e-2 / 4  # exp 6

MOVEMENT = 10.0  # exp 1, 2, 3, 4, 5
# MOVEMENT = 1.0  # exp 6

trainer = Trainer(
    z_samples=z_samples,
    movement=MOVEMENT,
    model=stochastic_ffnn,
    learning_rate=LEARNING_RATE,
    milestones=[
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
    ],  # exp 1, 2, 3, 4, 5, 6
    gamma=0.5,  # exp 1, 2, 3, 4, 5
    # gamma=0.85,  # exp 6
    device=device,
)

plotter = Plotter(
    datasets=datasets,
    z_samples_size=z_samples.z_samples.shape[0],
    test_s=0.9,
    train_s=0.2,
    zline_s=5,
    zline_skip=TEST_SIZE // 50,
    zsample_label=True,
)

tester = Tester(
    z_samples=z_samples,
    datasets=datasets,
    trainer=trainer,
    plotter=plotter,
    model=stochastic_ffnn,
    device=device,
)

# %%

# The total number of epochs to run training for.
NUM_EPOCHS = 181  # exp 1, 2
# NUM_EPOCHS = 260  # exp 3
# NUM_EPOCHS = 320  # exp 4, 5
# NUM_EPOCHS = 401  # exp 6

for epoch in range(0, NUM_EPOCHS):
    start = time.time()

    for x, y in datasets.data_loader_train:
        trainer.batch(
            x_pt=x, y_pt=y,
        )

    trainer.step(epoch=epoch)

    tester.step(epoch=epoch)

    end = time.time()
    print("epoch: {} elapsed time: {}".format(epoch, end - start))

# %%
