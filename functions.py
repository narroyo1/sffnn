"""
This module can be used to compose a stochastic function and train a neural network
to aproximate it.
"""

import numpy as np

from scipy.stats import truncnorm

# %%

###########################################
# 1D base functions
###########################################


def fn_sin(x_np, *, multiplier=1.0):
    """ Sine function. """
    return np.sin(x_np) * multiplier


def fn_double_sin(x_np, y_np, *, amplitude=1.0, longitude=0.5):
    """ Two interleaving sine functions. """
    updown = np.random.choice(a=[1.0, -1.0], size=x_np.shape, p=[0.5, 0.5])
    y_np += np.sin(x_np * longitude) * updown * amplitude
    return x_np, y_np


def fn_lin(x_np, *, multiplier=3.1416):
    """ Linear function """
    return x_np * multiplier


def fn_const(x_np, *, offset=0.0):
    """ Constant function. """
    return np.zeros(x_np.shape) + offset


def fn_x2(x_np, y_np, *, multiplier=5.0):
    """ x^2 function. """
    y_np += x_np * x_np * multiplier
    return x_np, y_np


def fn_x3_x2(x_np, y_np, *, multiplier1=1.0, multiplier2=8.0, offset=1.0):
    """ Third degree polynomial. """
    y_np += multiplier1 * x_np * x_np * x_np + multiplier2 * x_np * x_np + offset
    return x_np, y_np


def fn_branch(x_np, y_np):
    """ Branching function. """
    updown0 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])
    updown1 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])
    updown2 = np.random.choice(a=[1.0, -1.0], size=y_np.shape, p=[0.5, 0.5])

    y_np[x_np >= -2.0] += updown0[x_np >= -2.0]
    y_np[x_np >= 0.0] += updown1[x_np >= 0.0]
    y_np[x_np >= 2.0] += updown2[x_np >= 2.0]
    return x_np, y_np


###########################################
# 2D base functions
###########################################


def fn_x0_2_x1_2(x_np, y_np, *, multipler0=1, multiplier1=1):
    """ x0^2 + x1^2  """
    y_np += (
        x_np[:, 0] * x_np[:, 0] * multipler0
        + x_np[:, 1] * x_np[:, 1] * x_np[:, 1] * multiplier1
    )[..., np.newaxis]
    return x_np, y_np


def fn_x2_2d(x_np, y_np, *, multiplier=2.0):
    """ x^2 function. """
    # y_np += x_np * x_np * multiplier
    y_np[:, 0] += x_np[:, 0] * x_np[:, 0] * multiplier
    return x_np, y_np


def fn_circle(x_np, y_np, *, radius=10.0):
    length = np.sqrt(np.random.uniform(0, radius, (x_np.shape[0],)))
    # length = np.random.uniform(0, radius, (x_np.shape[0],))
    angle = np.pi * np.random.uniform(0, 2, (x_np.shape[0],))

    # result = np.zeros((x_np.shape[0], 2))
    y_np[:, 0] += length * np.cos(angle)
    y_np[:, 1] += length * np.sin(angle) * 0.5

    return x_np, y_np

    # Circle
    result = np.random.rand(x_np.shape[0] * 2, 2) * np.array([radius, radius])
    in_hypersphere = np.sum(result * result, axis=1) <= radius ** 2
    result = result[in_hypersphere]

    return result[: x_np.shape[0]]


def fn_rectangle(x_np, y_np, side1=3.0, side2=3.0):
    # Rectangle
    y_np = np.random.rand(x_np.shape[0], 2) * np.array([side1, side2])
    # result[:, 1] += result[:, 0]

    return x_np, y_np


def fn_2out_linear(x_np, multiplier1=1.0, multiplier2=1.0):
    result = np.zeros((x_np.shape[0], 2))
    for i in range(x_np.shape[0]):
        result[i, 0] = np.random.random()  # + x_np[i]
        result[i, 1] = result[i, 0] + (np.random.random() - 0.5)
    # return x_np * 0.0001 * np.array([multiplier1, multiplier2])
    return result


###########################################
# noise functions
###########################################


def fn_normal(x_np, y_np, *, std=1.2):
    """ Normal distribution noise. """
    y_np += np.random.randn(y_np.shape[0], 1) * std
    return x_np, y_np


def fn_exponential(x_np, *, scale=1.2):
    """ Exponential distribution noise. """
    return np.random.exponential(scale, x_np.shape[0])[..., np.newaxis]


def fn_sinnormal(x_np, y_np, *, amplitude=1.0, longitude=0.2, offset=0.0):
    """ Normal distribution noise multipled by sine function. """
    y_np += (
        np.sin((x_np + offset) * longitude)
        * np.random.randn(x_np.shape[0], 1)
        * amplitude
    )
    return x_np, y_np


def fn_halfnormal(x_np, y_np, *, std=1.2):
    """ Normal distribution noise truncated by half. """
    y_np += abs(np.random.randn(x_np.shape[0], 1)) * std
    return x_np, y_np


def fn_invertednormal(x_np, *, std=1.2, separation=1.2):
    """ Normal distribution noise growing in density towards the edges. """
    result = np.random.randn(*x_np.shape) * std
    result[result >= 0.0] = separation - result[result >= 0.0]
    result[result < 0.0] = -separation - result[result < 0.0]

    return result


def fn_normalx(x_np, *, std=5.0):
    """ Normal distribution noise multipled by the value of "x". """
    return np.random.randn(*x_np.shape) * x_np * std


def fn_noop(x_np, y_np):
    return x_np, y_np


def fn_normal2d(x_np, y_np, *, std=5.0):
    """ Normal distribution noise multipled by the value of "x". """
    mean = [0, 0]
    cov = [[std, 0], [0, std]]
    y_np += np.random.multivariate_normal(mean, cov, x_np.shape[0])
    return x_np, y_np

    # c = np.random.rand(x_np.shape[0])
    # n = np.random.rand(x_np.shape[0])
    x1 = np.random.rand(x_np.shape[0])[:, np.newaxis] * std
    x2 = np.random.rand(x_np.shape[0])[:, np.newaxis] * std
    return np.concatenate((x1, x2), axis=1)
    return np.concatenate((x1, x2 + x1), axis=1)
    # return np.random.rand(x_np.shape[0], 2) * std


def fn_uniform(x_np, *, multiplier=1.0):
    """ Uniform distribution noise. """
    return np.random.rand(*x_np.shape) * multiplier


def fn_double_normal(x_np, *, std=0.5, separation=1.7):
    """ Two normal distributions noise separated from each other. """
    updown = np.random.choice(
        a=[-separation / 2, separation / 2], size=(x_np.shape[0], 1), p=[0.5, 0.5]
    )
    y_np = np.random.randn(x_np.shape[0], 1) * std + separation * updown
    return y_np


def fn_truncnormal(x_np, y_np, *, mean=0, std=20, low=-50, upp=50):
    """ Truncated normal distribution noise. """
    x_trunc = truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=std)
    y_np += x_trunc.rvs(x_np.shape[0])[..., np.newaxis]
    return x_np, y_np


def binder(func, x_space_size=1, y_space_size=1, **kwargs):
    """
    This function binds named arguments to a function taking 1 numpy array positional argument.
    """

    def helper(x_np, y_np):
        return func(x_np, y_np, **kwargs)

    helper.x_space_size = x_space_size
    helper.y_space_size = y_space_size
    helper.name = func.__name__

    return helper
