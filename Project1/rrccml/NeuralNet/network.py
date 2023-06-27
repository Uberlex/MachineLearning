# overview of network class:
#     Att: linked list of layers
#     Methods: add_layer, add_precessing, compile and train (diagnostic)
#         Output while training. [Training Score: 0.797 , Training Loss: 0.140 ] [Val Score: 0.695 , Val Loss: 0.179 ]
#         Should save the model. (while training and at the end)
#             - json with path to weights.
#         Load pretrained weights from log by timestamp.
from rrccml.Neural_Net import layer as lyr
import numpy as np


class Network:
    """A neural network.

    Attributes
    ----------
    depth : int
        number of layers the network has
    processes : list
        collection of processes to run on the features data set before training / running
    layers : list
        the layers of the network in order
    features : np.array
        features on which to train the network
    targets : np.array
        target classifications
    processed_data : np.array
        if the processing layers were run (and `store` = True), this will be populated with the result
    history : list
        a summary of the error at each epoch

    Methods
    -------
    add_process :
        adds a processing function to the network
    run_processes :
    get_weights : list
        helper function to get the weights of each layer
    forward_pass : np.array
        orchestrates each layers forward_pass method
    backward_pass :
        orchestrates each layers backward_pass method
    add_layer :
        adds a layer to the network
    compile :
        runs an integrity check on the network to ensure that the layers mesh
    train : None
        trains the network, saving weights periodically
    run :
        runs the trained network on unseen features.
    """

    def __init__(self, features, targets):
        self.depth = 0
        self.processes = []
        self.layers = []
        self.features = features
        self.targets = targets
        self.processed_data = None
        self.history = []

    def add_process(self, f, **kwargs):
        """Adds a processing function into the processing steps. Processing functions alter / transform the features.
        Processing functions first argument should expect to accept `features`

        Example structure f(data, x=1, y=2)

        Parameters
        ----------
        f : function
            A function ready to accept either the features data or the result of the previous process.
        **kwargs : **dict
            Keyword arguments to pass to `f`
        """
        process = {'function': f, 'kwargs': kwargs}
        self.processes.append(process)

    def run_processes(self, data=None, store=True):
        """Runs the sequence of processes on the provided data
        Parameters
        ----------
        data : np.array / pd.DataFrame
            The data on which to begin processing
        store : bool
            Whether to save the result in the `processed_data` attribute

        Returns
        -------
        data
            The result of running each process in the process list
        """
        if data is None:
            data = self.features

        for process in self.processes:
            f = process.get("function")
            kwargs = process.get("kwargs")
            data = f(data, **kwargs)

        if store:
            self.processed_data = data

        return data

    def add_layer(self, layer):
        """Add a layer to the network
        Parameters
        ----------
        layer : Layer
            An instance of class Layer
        """
        # Checks whether the provided layer is of the appropriate class
        if not isinstance(layer, lyr.Layer):
            raise TypeError("layer must be of type `Layer`")

        # Captures the number of columns from previous layer
        if not self.layers:
            row_count = self.features.shape[1]
        else:
            row_count = self.layers[-1].width

        # Sets the layers row attribute to be compatible with previous weight matrix
        layer.set_rows(row_count)
        layer.layer_index = self.depth + 1

        # Add the layer to the network
        self.layers.append(layer)

        # Increase the depth of the network by 1
        self.depth += 1

    def compile(self):
        """Checks compatibility across each layer and initializes random weights"""
        for layer in self.layers:
            layer.compile()

    def get_weights(self):
        """Helper function to access weights of each layer"""
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)

        return weights

    def forward_pass(self, data):
        # Forward Pass
        for layer in self.layers:
            data = layer.forward_pass(data)

        return data

    def backward_pass(self, error):
        # Backward Pass
        for layer in reversed(self.layers):
            error = layer.backward_pass(error)

    def train(self, x_val, y_val, epochs, cost_function, eval_function, learning_rate=.01):
        """Trains the network using features
        Parameters
        ----------
        epochs : int
            The number of times to loop throught the training process
        x_val : np.array
            The validation features
        y_val : np.array
            The validation targets
        cost_function : function
            The cost function to measure distance from actual.
            Should be ready to accept two arguments: actual and expected values.
        eval_function : function
            Function to measure evaluate the results.
            Should be ready to accept two arguments: features and targets values.
        learning_rate : float
            Size of steps to take down the gradient
        """
        data = self.features if self.processed_data is None else self.processed_data

        for epoch in range(epochs):

            # Forward Pass
            current_a = self.forward_pass(data)

            # Evaluate the Error
            trained_error = cost_function(current_a, self.targets)

            # Backward Pass
            self.backward_pass(trained_error)

            # Update Weights
            # Could be wrapped into a method
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z = data

                layer.update_weights(learning_rate, z)
                z = layer.activation.function(layer.z)

            self.history.append([epoch, np.array(np.sum((current_a - self.targets) ** 2) / np.shape(self.targets)[1])[0]])

            # Validation PSUEDO CODE
            # training_score = eval_function(self.features, self.targets, )
            # validation_score = eval_function(x_val, y_val)
            # FPV = self.forward_pass(x_val)
            # EV = cost_function(y_val, FPV)

        # Temporarily just returns the result of forward passes
        return current_a

    def run(self, features, process=True):
        """Runs new features against current matrices
        Parameters
        ----------
        features : np.array
            New features to run through the network
        process : bool
            Whether to should run the new features through the processing layers

        Returns
        -------
        results : np.array
            The resulting matrix of the output layer
        """

        if process:
            # Transform the data using the process layers
            data = self.run_processes(data=features, store=False)
        else:
            # Initialize running results
            data = features

        # Feed results through each layer...
        result = self.forward_pass(data)

        return result
