import numpy as np
from rrccml.Neural_Net.activation import activation_functions


class Layer:
    """Represents a base class for layers in neural network. Intended for inheritence, not direct use.

    Attributes
    ----------
    weights : numpy array (matrix) of floats
        all the weights associated to the layer
    activation : vector of floats
        all the results prior to activation for use in back propogation
    activation_function : Activation_Function object
        the activation function

    Methods
    -------
    forward_pass : a
        pushing data from the previous layer to the next
    backward_pass : None
        adjusts weights during training
    """
    def __init__(self):
        self.weights = None
        self.activation = None


class Connected(Layer):
    """A typical feed-forward layer

    Attributes
    ----------
    width : int
        Number of nodes the layer has
    activation : str
        keyword related to the desired activation function
    weights : np.array
        the strength of association between the previous layer
    rows : int
        number of columns the previous layer has (or the number of rows this layers weight matrix should have)
    z : np.array
        For a given epoch, the resulting z (before activation)
    delta : np.array
        stored difference between error and derivative of activation function of z for use in updating the weights
    """
    def __init__(self, width, activation="sigmoid", weights=None):

        # The number of nodes and/or width of the given layer
        self.width = width

        # Number of rows the weight matrix should have. Set by `Network` on Network.add_layer()
        self.rows = 0

        # Select the activation function for this specific layer
        self.activation = activation_functions.get(activation)

        # Initialize weights if any
        self.weights = weights

        # Need a better name. Stores the z during each forward pass for use in back propagation
        self.z = None
        self.delta = None
        self.layer_index = None

    def set_rows(self, r):
        self.rows = r

    def forward_pass(self, a):
        # Stores the z during each forward pass for use in back propagation
        self.z = np.dot(a, self.weights)
        a = self.activation.function(self.z)
        return a

    def compile(self):
        """What happens when Network calls compile"""
        if self.weights is None:
            # Generate Random Weights
            self.weights = 2 * np.random.rand(self.rows, self.width) - 1

        # This checks to see if the custom weights are compatible with the previous layers matrix
        elif self.weights.shape != (self.rows, self.width):
            # Need better error message
            raise ValueError("Weight shape is not compatible with rows and width?")

    def backward_pass(self, error):
        self.delta = error * self.activation.derivative(self.z)
        return np.dot(self.delta, self.weights.T)

    def update_weights(self, learning_rate, z):
        """Updates a layers weights using the learning rate and the z from the previous layer
        Paramters
        ---------
        learning_rate : float
            How big of steps to take down the gradient
        z : np.array
            The z from the previous layer (or the networks features if first layer)
        """
        self.weights -= learning_rate * np.dot(z.T, self.delta)
