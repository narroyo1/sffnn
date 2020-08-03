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


from scipy.stats import truncnorm

from model import StochasticFFNN
from trainer import Trainer
from tester import Tester
from zsamples import ZSamples
from datasets import DataSets


# %%

###########################################
# 1D base functions
###########################################


def fn_sin(x_np, multiplier=1.0):
    """ Sine function. """
    return np.sin(x_np) * multiplier


def fn_double_sin(x_np, amplitude=1.0, longitude=0.5):
    """ Two interleaving sine functions. """
    updown = np.random.choice(a=[1.0, -1.0], size=x_np.shape, p=[0.5, 0.5])
    return np.sin(x_np * longitude) * updown * amplitude


def fn_lin(x_np, multiplier=3.1416):
    """ Linear function """
    return x_np * multiplier


def fn_const(x_np, offset=0.0):
    """ Constant function. """
    return np.zeros(x_np.shape) + offset


def fn_x2(x_np, multiplier=5.0):
    """ x^2 function. """
    return x_np * x_np * multiplier


def fn_x3_x2(x_np, multiplier1=1.0, multiplier2=8.0, offset=1.0):
    """ Third degree polynomial. """
    return multiplier1 * x_np * x_np * x_np + multiplier2 * x_np * x_np + offset


def fn_branch(x_np):
    """ Branching function. """
    y_np = np.zeros(x_np.shape)
    updown0 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])
    updown1 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])
    updown2 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])

    y_np[x_np >= -2.0] += updown0[x_np >= -2.0]
    y_np[x_np >= 0.0] += updown1[x_np >= 0.0]
    y_np[x_np >= 2.0] += updown2[x_np >= 2.0]
    return y_np


###########################################
# 2D base functions
###########################################


def fn_x0_2_x1_2(x_np, multipler0=1, multiplier1=1):
    """ x0^2 + x1^2  """
    result = (
        x_np[:, 0] * x_np[:, 0] * multipler0
        + x_np[:, 1] * x_np[:, 1] * x_np[:, 1] * multiplier1
    )
    return result[..., np.newaxis]


###########################################
# noise functions
###########################################


def fn_normal(x_np, std=1.2):
    """ Normal distribution noise. """
    return np.random.randn(x_np.shape[0], 1) * std


def fn_exponential(x_np, scale=1.2):
    """ Exponential distribution noise. """
    return np.random.exponential(scale, x_np.shape[0])[..., np.newaxis]


def fn_sinnormal(x_np, amplitude=1.0, longitude=0.2, offset=0.0):
    """ Normal distribution noise multipled by sine function. """
    return (
        np.sin((x_np + offset) * longitude)
        * np.random.randn(x_np.shape[0], 1)
        * amplitude
    )


def fn_halfnormal(x_np, std=1.2):
    """ Normal distribution noise truncated by half. """
    return abs(np.random.randn(x_np.shape[0], 1)) * std


def fn_invertednormal(x_np, std=1.2, separation=1.2):
    """ Normal distribution noise growing in density towards the edges. """
    result = np.random.randn(*x_np.shape) * std
    result[result >= 0.0] = separation - result[result >= 0.0]
    result[result < 0.0] = -separation - result[result < 0.0]

    return result


def fn_normalx(x_np, multiplier=5):
    """ Normal distribution noise multipled by the value of "x". """
    return np.random.randn(*x_np.shape) * x_np * multiplier


def fn_uniform(x_np, multiplier=1.0):
    """ Uniform distribution noise. """
    return np.random.rand(*x_np.shape) * multiplier


def fn_double_normal(x_np, std=0.5, separation=1.7):
    """ Two normal distributions noise separated from each other. """
    updown = np.random.choice(
        a=[-separation / 2, separation / 2], size=(x_np.shape[0], 1), p=[0.5, 0.5]
    )
    y_np = np.random.randn(x_np.shape[0], 1) * std + separation * updown
    return y_np


def fn_truncnormal(x_np, mean=0, std=20, low=-50, upp=50):
    """ Truncated normal distribution noise. """
    x_trunc = truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=std)
    return x_trunc.rvs(x_np.shape[0])[..., np.newaxis]


def binder(func, x_space_size=1, **kwargs):
    """
    This function binds named arguments to a function taking 1 numpy array positional argument.
    """

    def helper(x_np):
        return func(x_np, **kwargs)

    helper.x_space_size = x_space_size
    helper.name = func.__name__

    return helper


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
datasets.show()


NUM_Z_SAMPLES = 13  # exp 1, 2, 5
# NUM_Z_SAMPLES = 18  # exp 3, 4
# NUM_Z_SAMPLES = 6  # exp 6
# The range of values in z-space.
# Having a range spanning from negative to positive can have unintended
# training results.
Z_RANGE = np.array([[-10.0, 10.0]])  # exp 1, 2, 3, 4, 5
# Z_RANGE = np.array([[10.0, 20.0]])  # exp 6

z_samples = ZSamples(num_z_samples=NUM_Z_SAMPLES, z_range=Z_RANGE, device=device)


# %%

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

tester = Tester(
    z_samples=z_samples,
    datasets=datasets,
    trainer=trainer,
    model=stochastic_ffnn,
    device=device,
    test_s=0.9,
    train_s=0.2,
    zline_s=5,
    zline_skip=TEST_SIZE // 50,
    zsample_label=True,
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
