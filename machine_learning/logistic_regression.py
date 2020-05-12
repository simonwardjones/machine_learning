import json

import numpy as np


class LogisticRegression():

    def __init__(self, learning_rate=0.05):
        """  
        Logistic regression model

        Parameters:
        ----------
        learning_rate: float, optional, default 0.05
            The learning rate parameter controlling the gradient descent
            step size
        """
        self.learning_rate = learning_rate
        print('Creating logistic model instance')

    def __repr__(self):
        return (
            f'<LogisticRegression '
            f'learning_rate={self.learning_rate}>')

    def fit(self, X, y, n_iter=1000):
        """  
        Fit the logistic regression model

        Updates the weights with n_iter iterations of batch gradient
        descent updates

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)
        y: numpy.ndarray
            Target values - class label {0, 1}, shape (m samples, 1)
        n_iter: int, optional, default 1000
            Number of batch gradient descent steps
        """
        m, n = X.shape
        print(f'fitting with m={m} samples with n={n-1} features\n')
        self.beta = np.zeros(shape=(n, 1))
        self.costs = []
        self.betas = [self.beta]
        for iteration in range(n_iter):
            y_pred = self.predict_proba(X)
            cost = (-1 / m) * (
                (y.T @ np.log(y_pred)) +
                ((np.ones(shape=y.shape) - y).T @ np.log(
                    np.ones(shape=y_pred.shape) - y_pred))
            )
            self.costs.append(cost[0][0])
            gradient = (1 / m) * X.T @ (y_pred - y)
            self.beta = self.beta - (
                self.learning_rate * gradient)
            self.betas.append(self.beta)

    def predict_proba(self, X):
        """  
        Predicted probability values for class 1

        Note this is calculated as the sigmoid of the linear combination
        of the feature values and the weights.

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)

        Returns:
        -------
        numpy.ndarray:
            Predicted probability of samples being in class 1
        """        
        y_pred = self.sigmoid(X @ self.beta)
        return y_pred

    def predict(self, X, descision_prob=0.5):
        """  
        Predict the class values from sample X feature values

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, (n - 1) features + 1)
            Note the first column of X is expected to be ones (to allow 
            for the bias to be included in beta)

        Returns:
        -------
        numpy.ndarray:
            Prediceted class values, shape (m samples, 1)
        """
        y_pred = self.sigmoid(X @ self.beta)
        return (y_pred > descision_prob) * 1

    def sigmoid(self, x):
        """  
        Sigmoid function

        f(x) = 1 / (1 + e^(-x))

        Parameters:
        ----------
        x: numpy.ndarray

        Returns:
        -------
        numpy.ndarray:
            sigmoid of x, values in (0, 1)
        """        
        return 1 / (1 + np.exp(-x))
