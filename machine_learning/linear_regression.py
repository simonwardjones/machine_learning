import json

import numpy as np


class LinearRegression():

    def __init__(self, learning_rate=0.05):
        """
        Linear regression model

        Parameters:
        ----------
        learning_rate: float, optional, default 0.05
            The learning rate parameter controlling the gradient descent
            step size
        """
        self.learning_rate = learning_rate
        print('Creating linear model instance')

    def __repr__(self):
        return (
            f'<LinearRegression '
            f'learning_rate={self.learning_rate}>')

    def fit(self, X, y, n_iter=1000):
        """
        Fit the linear regression model

        Updates the weights with n_iter iterations of batch gradient
        descent updates

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)
        y: numpy.ndarray
            Target values, shape (m samples, 1)
        n_iter: int, optional, default 1000
            Number of batch gradient descent steps
        """
        m, n = X.shape
        print(f'fitting with m={m} samples with n={n-1} features\n')
        self.beta = np.zeros(shape=(n, 1))
        self.costs = []
        self.betas = [self.beta]
        for iteration in range(n_iter):
            y_pred = self.predict(X)
            cost = self.cost(y, y_pred)
            self.costs.append(cost[0][0])
            gradient = self.gradient(y, y_pred, X)
            self.beta = self.beta - (
                self.learning_rate * gradient)
            self.betas.append(self.beta)

    def cost(self, y, y_pred):
        """  
        Mean square error cost function

        Parameters:
        ----------
        y: numpy.ndarray
            True target values, shape (m samples, 1)
        y_pred: numpy.ndarray
            Predicted y values, shape (m samples, 1)

        Returns:
        -------
        float:
            mean square error value
        """
        m = y.shape[0]
        cost = (1 / (2 * m)) * (y - y_pred).T @ (y - y_pred)
        return cost

    def gradient(self, y, y_pred, X):
        """  
        Calculates the gradient of the cost function

        Parameters:
        ----------
        y: numpy.ndarray
            Predicted y values, shape (m samples, 1)
        y_pred: numpy.ndarray
            True target values, shape (m samples, 1)
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)

        Returns:
        -------
        numpy.ndarray:
            Derivate of mean square error cost function with respect to
            the weights beta, shape (n features, 1)
        """
        m = X.shape[0]
        gradient = (1 / m) * X.T @ (y_pred - y)
        return gradient

    def predict(self, X):
        """  
        Predict the target values from sample X feature values

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)

        Returns:
        -------
        numpy.ndarray:
            Target value predictions, shape (m samples, 1)
        """
        y_pred = X @ self.beta
        return y_pred
