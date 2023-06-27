"""Provides a dictionary of activation functions called "activation_functions"
"""
import numpy as np


class Activation_Function:
    """The wrapper for activation functions.
    Attributes
    ----------
    function : function
        the function itself, used for forward passing
    derivative : function
        the derivative of the function, used for back propogation
    """

    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def s(z):
    return 1 / (1 + np.e ** (-z))


def s_d(z):
    return s(z) * (1 - s(z))


# This dictionary will be the primary export of this module.
activation_functions = {
    "sigmoid": Activation_Function(s, s_d)
}
