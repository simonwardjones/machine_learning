import logging

import numpy as np

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class NeuralNetwork():

    def __init__(self,
                 layer_sizes=[5, 10, 1],
                 is_classifier=True,
                 learning_rate=0.1):
        """Neural network model

        Parameters:
        ----------
        layer_sizes: list, optional, default [5, 10, 1]
            Number of nodes in each layer (including input and output)
        is_classifier: bool, optional, default True
            Is the model used as part of a classification problem
            or a regression problem. Should be set to True if
            classification, False if regression
        learning_rate: float, optional, default 0.05
            The learning rate parameter controlling the gradient descent
            step size
        """
        self.layer_sizes = layer_sizes  # n^0, ..., n^L
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.n_L = layer_sizes[-1]  # n^L
        self.n_layers = len(layer_sizes) - 1  # L
        self.initialise_weights()

    def initialise_weights(self):
        """Initialise the weights and biases

        weights are initialized as small random numbers, biases as zero
        """
        self.weight_matrices = [
            np.random.normal(loc=0.0, scale=1.0, size=(n_l, n_l_minus_1))
            for n_l, n_l_minus_1 in zip(self.layer_sizes[1:], self.layer_sizes)
        ]
        self.betas = [np.zeros(shape=(n_l, 1)) for n_l in self.layer_sizes[1:]]

    def feed_forward(self, X):
        """Feed X forward through the network

        For each layer the net input is calculated as the product of the
        weight matrix and the activations of the previous layer plus the
        biases.

        The output activation is then calculated by applying the
        activation function to the net input

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            final layer activations, shape (n^L, m)
        """
        m = X.shape[0]
        layer_activations = [X.T]
        for layer in range(self.n_layers):
            A_layer_minus_1 = layer_activations[-1]
            beta = self.betas[layer]
            B = np.repeat(beta, m, axis=-1)
            Z = self.weight_matrices[layer] @ A_layer_minus_1 + B
            A = self.activation_function(Z, layer=layer)
            layer_activations.append(A)
            self.log_layer(layer, A_layer_minus_1, beta, B, Z, A)
        self.layer_activations = layer_activations
        return layer_activations[-1]

    def back_propagation(self, X, Y):
        """Update the weights and biases through back propagation

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        Y: numpy.ndarray
            Target values, shape (n_classes, m samples)
        """
        assert X.shape[0] == Y.shape[1]
        final_layer_error = self.layer_activations[-1] - Y
        D_plus_1 = final_layer_error
        # errors represent D matrices in notebook explanation
        errors = [D_plus_1]
        for layer in range(self.n_layers - 2, -1, -1):
            logger.debug(f'Calculating D_{layer + 1}')
            A = self.layer_activations[layer + 1]
            self.log_back_prop_layer(layer, A, D_plus_1)
            D = (self.weight_matrices[layer + 1].T @ D_plus_1) * \
                A * (1 - A)
            D_plus_1 = D
            errors.insert(0, D)
        self.errors = errors
        self.update_weights()

    def update_weights(self):
        """Update the weights and biases using gradient

        The weights and biases are updated by calculating the parital
        derivatives and then stepping the weights in the direction 
        of the negative gradient. The step size is governed by the
        learing rate
        """
        for layer in range(self.n_layers):
            m = self.errors[0].shape[1]
            d_L_d_W = (1 / m) * self.errors[layer] @ \
                self.layer_activations[layer].T
            d_L_d_beta = (1 / m) * self.errors[layer].sum(axis=1)[:, None]
            self.weight_matrices[layer] = self.weight_matrices[layer] - \
                self.learning_rate * d_L_d_W
            if layer == 0:
                self.d_L_d_Ws.append(d_L_d_W.sum())
            self.betas[layer] = self.betas[layer] - \
                self.learning_rate * d_L_d_beta

    def log_layer(self, layer, A_layer_minus_1, beta, B, Z, A):
        """Utility function to group logging

        Parameters:
        ----------
        layer: int
            The layer being logged (note python uses 0 index) so the
            layer is actually layer + 1
        A_layer_minus_1: numpy.ndarray, shape (n^{l-1},m)
            Previous layer activations for each sample
        beta: numpy.ndarray, shape (n^{l}, 1)
            Layer biases
        B: numpy.ndarray, shape (n^{l}, m)
            Repeated layer biases for ease of matrix operations
        Z: numpy.ndarray, shape (n^{l}, m)
            Net input for each sample
        A: numpy.ndarray, shape (n^{l}, m)
            Output activation for each sample
        """
        logger.debug(
            f'A_layer_minus_1 i.e. A_{layer} '
            f'has shape {A_layer_minus_1.shape}')
        logger.debug(f'beta_{layer + 1} has shape {beta.shape}')
        logger.debug(f'B_{layer + 1} has shape {B.shape}')
        logger.debug(f'Z_{layer + 1} has shape {Z.shape}')
        logger.debug(f'A_{layer + 1} has shape {A.shape}')

    def log_back_prop_layer(self, layer,  A, D_plus_1):
        """Utility for logging back propagation

        Parameters:
        ----------
        layer: int
            The layer being logged (note python uses 0 index) so the
            layer is actually layer + 1
        A: numpy.ndarray, shape (n^{l}, m)
            Output activation for each sample
        D_plus_1: numpy.ndarray, shape (n^{l+1}, m)
            Error in the next layer
        """
        logger.debug(
            f'A_{layer + 1} has shape {A.shape}')
        logger.debug(
            f'W_{layer + 2} has shape '
            f'{self.weight_matrices[layer + 1].shape}')
        logger.debug(
            f'D_{layer + 2} has shape {D_plus_1.shape}')

    def activation_function(self, Z, layer):
        """Activation function

        The activation function is the sigmoid for nodes except the
        output layer. For the final layer the identify function is used
        for regression and for multiclass classification the softmax
        function is used

        Parameters:
        ----------
        Z: numpy.ndarray, shape (n^{l}, m)
            Net input for each sample
        layer: int
            The layer being logged (note python uses 0 index) so the
            layer is actually layer + 1

        Returns:
        -------
        numpy.ndarray:
            Output activation for each sample, shape (n^{l}, m)
        """
        if layer == (self.n_layers - 1):
            if not self.is_classifier:
                return Z
            if self.is_classifier and self.n_L >= 2:
                return np.exp(Z - logsumexp(Z, axis=0)[None, :])
        return expit(Z)

    def cost(self, Y):
        """Cost function

        Parameters:
        ----------
        Y: numpy.ndarray
            Target values, shape (n_classes, m samples)
        """
        if self.is_classifier and self.n_L == 1:
            cost = (-1 / m) * (
                Y * np.log(self.layer_activations[-1]) +
                (1 - Y) * np.log(1 - self.layer_activations[-1])
            ).sum()
        if self.is_classifier and self.n_L > 1:
            cost = (-1 / m) * \
                (Y * np.log(self.layer_activations[-1])).sum()
        if not self.is_classifier:
            cost = (1 / (2 * m)) * \
                ((Y - self.layer_activations[-1]) ** 2).sum()
        logger.debug(f'cost = {cost}')
        self.costs.append(cost)

    def fit(self, X, Y, epochs=100):
        """Fits the neural network with training data

        The fitting is done via multiple epochs of gradient descent.
        Each iteration has a feed forward step and a back propagation
        step.

        Note Y is one hot encoded if necessary.

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        Y: numpy.ndarray
            Target values, shape (m samples, 1)
        epochs: int, optional, default 100
            Number of iterations of gradient descent
        """
        if self.n_L > 1:
            if Y.shape[0] != self.n_L:
                print('One hot encoding Y')
                Y = np.eye(self.n_L)[:, Y.reshape(-1).astype(int)]
        self.costs = []
        self.d_L_d_Ws = []
        for epoch in range(epochs):
            self.feed_forward(X)
            self.cost(Y)
            self.back_propagation(X, Y)

    def predict(self, X):
        """Predicts target values or class labels by forward propagation

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        Returns:
        -------
        numpy.ndarray:
            Predicted target values or class labels for classification,
            Shape is (n^L, m samples)
        """
        A_L = self.feed_forward(X)
        if not self.is_classifier:
            return A_L
        if self.is_classifier and self.n_L == 1:
            return np.round(A_L).astype(int)
        if self.is_classifier and self.n_L > 1:
            return np.argmax(A_L, axis=0)

    def predict_proba(self, X):
        """Predicts class probabilities for input data

        Parameters:
        ----------
        X: numpy.ndarray
            Sample data, shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted sample class probabilities, 
            shape (n classes, m samples)
            if n_classes > 2 else shape (1, m samples)
        """
        A_L = self.feed_forward(X)
        if not self.is_classifier:
            raise Exception('Must be a classifier')
        return A_L
