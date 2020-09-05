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


def fn_2out_linear(x_np, multiplier=1.0):
    return x_np * np.array([multiplier, multiplier])


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

    # helper.x_space_size = x_space_size
    helper.name = func.__name__

    return helper
