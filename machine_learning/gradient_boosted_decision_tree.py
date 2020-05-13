import logging

import numpy as np
from scipy.special import expit, logsumexp

from .decision_tree import DecisionTree

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class GradientBoostedDecisionTree():

    def __init__(self,
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_classes=2,
                 max_features=None,
                 is_classifier=True,
                 n_trees=10,
                 learning_rate=0.1):
        """Gradient boosted decision tree model

        The trees are grown sequentially and fitted to the negative 
        gradient of the cost function with respect to the raw predicted
        values at the previous stage. 

        Note the I use the term raw_predictions as raw predicted values 
        must be transformed to find the probability estimates in the 
        case of classification.

        In practice this gradients are equal to the residual.

        The raw predictions for a stage are made by adding the new delta
        model (multiplied by the learning rate) to the raw predictions
        from the previous stage

        Parameters:
        ----------
        max_depth: int
            The maximum depth allowed when "growing" a tree
        min_samples_split: int
            The minimum number of samples required to allow a split at a
            node
        min_samples_leaf: int
            The minimum number of samples allowed in a leaf. A split
            candidate leading to less samples in a node than the
            min_samples_leaf will be rejected
        n_classes: int, optional, default 2
            Number of classes in a classification setting. Ignored when
            self.is_classifier = False
        max_features: int, optional, default None
            If set to 'sqrt' then only a random subset of features are
            used to split at each node, the number of features used in
            this case is sqrt(n_features).
            Else all the features are considered when splitting at each
            node
        is_classifier: bool, optional, default True
            Is the model used as part of a classification problem
            or a regression problem. Should be set to True if
            classification, False if regression
        n_trees: int, optional, default 10
            Number of trees in the forest
        learning_rate: float, optional, default 0.05
            The learning rate parameter controlling the gradient descent
            step size
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.max_features = max_features
        self.is_classifier = is_classifier

        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.is_fitted = False
        np.random.seed(1)
        self.trees_to_fit = 1 if n_classes <= 2 else n_classes
        self.trees = [
            [None for _ in range(self.trees_to_fit)]
            for _ in range(self.n_trees)]
        #  trees has shape (n_trees, n_classes)

    def predict_delta_model(self, X, stage=0):
        """Calculate the delta model for a stage

        This function returns the estimate of the negative gradient. 
        These raw predictions are the delta models f_{stage + 1}

        Parameters:
        ----------
        X: numpy.ndarray
            Sample data, shape (m samples, n features)
        stage: int, optional, default 0
            What correction step are we predicting

        Returns:
        -------
        numpy.ndarray:
            gradient_step, shape (X.shape[0], n_classes)
            if n_classes > 2 else shape (m samples, 1)
        """
        class_gradient_step = []
        for class_k, model in enumerate(self.trees[stage]):
            k_gradient_step = model.predict(X).reshape(-1)
            class_gradient_step.append(k_gradient_step)
        gradient_step = np.stack(class_gradient_step, axis=-1)
        return gradient_step

    def predict_raw_stages(self, X, n_stages=None):
        """Predictions for input X

        The predictions are given by the transformed sum of initial 
        model and delta models. Note no transformation is required for
        regression.

        If n_stages specified stop at that stage. The delta model is
        multiplied by the learning rate before being added to the
        raw predictions

        Parameters:
        ----------
        X: numpy.ndarray
            Sample data, shape (m samples, n features)
        n_stages: in, optional, default None
            If given return prediction an n_stages

        Returns:
        -------
        numpy.ndarray:
            predictions, shape (X.shape[0], n_classes)
            if n_classes > 2 else shape (m samples, 1)
        """
        if not n_stages:
            n_stages = self.n_trees
        if n_stages not in list(range(1, self.n_trees + 1)):
            raise Exception('n_stages must be between 1 and n_trees')
        raw_predictions = self.f_0_prediction(X)
        for stage in range(n_stages):
            stage_gradient_step = self.predict_delta_model(X, stage)
            raw_predictions += self.learning_rate * stage_gradient_step
        return self.convert_raw_predictions(raw_predictions)

    def predict(self, X):
        """Predicts target values or class labels for classification

        Parameters:
        ----------
        X: numpy.ndarray
            Sample data, shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted target values or class labels for classification
        """
        if not self.is_classifier:
            return self.predict_raw_stages(X)
        else:
            return np.argmax(self.predict_proba(X), axis=-1)

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
            shape (m samples, n classes)
            if n_classes > 2 else shape (m samples, 1)
        """
        if not self.is_classifier:
            raise Exception('Not a classifier')
        if self.n_classes == 2:
            prob_class_one = self.predict_raw_stages(X)
            return np.stack([1-prob_class_one, prob_class_one], axis=-1)
        if self.n_classes > 2:
            return self.predict_raw_stages(X)

    def convert_raw_predictions(self, raw_predictions):
        """Convert raw_predictions to probability if classifier

        This uses sigmoid if the are two classes - in which case we
        model the logit. Softmax function is used when there are more
        than two classes.

        Parameters:
        ----------
        raw_predictions: numpy.ndarray
            Raw predictions, shape (m samples, n classes)

        Returns:
        -------
        numpy.ndarray:
            target values or class probabilities for classification
        """
        if not self.is_classifier:
            return raw_predictions
        if self.is_classifier and self.n_classes == 2:
            return expit(raw_predictions)
        if self.is_classifier and self.n_classes > 2:
            return np.exp(
                raw_predictions - logsumexp(raw_predictions, axis=1)[:, None])

    def f_0_prediction(self, X):
        """Return initial raw_predictions for X

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            raw_predictions, shape (m samples, n classes)
            if n_classes > 2 else shape (m samples, 1)
        """
        n = X.shape[0]
        if not self.is_classifier:
            return self.regression_f_0_tree.predict(X).reshape(n, 1)
        if self.is_classifier and self.n_classes == 2:
            return np.repeat(self.f_0, n).reshape(n, 1)
        if self.is_classifier and self.n_classes > 2:
            return np.repeat(self.f_0, n, axis=0)

    def init_f_0(self, X, y):
        """Fit initial prediction model

        For regression this is simple fitting a first tree to the target
        values.

        For classification when we model the logit (in two class 
        scenario) we use the logit of the average probability in the
        training data.
        For the multi class case, where we model the log of each class
        probability as an additive model, we initialise the raw values
        as the log of the observed probability of that class.

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        y: numpy.ndarray
            Target values, shape (m samples, 1)
            If classifier with n_classes the values are assumed to be in
            0, ..., n-1
        """
        y = y.reshape(-1)
        if not self.is_classifier:
            self.regression_f_0_tree = self.get_tree()
            self.regression_f_0_tree.fit(X, y)
        if self.is_classifier and self.n_classes == 2:
            self.f_0 = np.log(y.sum() / (y.shape[0] - y.sum()))
        if self.is_classifier and self.n_classes > 2:
            self.f_0 = np.log(
                np.bincount(y, minlength=self.n_classes) / y.shape[0])[None, :]

    def get_tree(self):
        """Helper to return decision tree to be fitted

        Returns:
        -------
        DecisionTree:
            Regression tree
        """
        return DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_classes=self.n_classes,
            max_features=self.max_features,
            impurity='mse',
            is_classifier=False)

    def fit(self, X, y):
        """Fit the gradient boosted decision tree

        For each stage fit a tree to the negative gradient (for that
        class), then update the raw predictions using the learning rate
        and delta model.

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        y: numpy.ndarray
            Target values, shape (m samples, 1)
            If classifier with n_classes the values are assumed to be in
            0, ..., n-1
        """
        if self.is_classifier:
            y = y.astype(int)
        self.init_f_0(X, y)
        prev_stage_raw_predictions = self.f_0_prediction(X)
        for stage in range(self.n_trees):
            negative_gradient = self.negative_gradient(
                y, prev_stage_raw_predictions)
            self.fit_stage(X, negative_gradient, stage=stage)
            delta_model = self.predict_delta_model(X, stage=stage)
            prev_stage_raw_predictions = prev_stage_raw_predictions + \
                (self.learning_rate * delta_model)

    def fit_stage(self, X, negative_gradient, stage=0):
        """Fit a given stage

        For regression this is just fitting a single tree to the
        gradient. For classification we fit one tree for each class (
        unless there are only two classes when we can use just one)

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        negative_gradient: numpy.ndarray
            dL_dY^hat, shape (m samples, n features)
        stage: int, optional, default 0
            stage to fit
        """
        logger.info(f'Fitting stage {stage}')
        trees_to_fit = 1 if self.n_classes <= 2 else self.n_classes
        for class_k in range(trees_to_fit):
            target = negative_gradient[:, class_k]
            tree = self.get_tree()
            tree.fit(X, target)
            self.trees[stage][class_k] = tree

    def negative_gradient(self, y, prev_stage_raw_predictions):
        """Gradient of the loss function with res

        Parameters:
        ----------
        y: numpy.ndarray
            Target values, shape (m samples, 1)
            If classifier with n_classes the values are assumed to be in
            0, ..., n-1
        prev_stage_raw_predictions: numpy.ndarray
            raw_predictions, shape

        Returns:
        -------
        numpy.ndarray:
            negative gradient, shape (m samples, n classes)
            if n_classes > 2 else shape (m samples, 1)
        """
        if self.is_classifier and self.n_classes > 2:
            y = np.eye(self.n_classes)[y.reshape(-1)]
        else:
            y = y.reshape(y.shape[0], 1)
        return y - self.convert_raw_predictions(prev_stage_raw_predictions)

    def render(self, stage, class_k, feature_names):
        """Returns Digraph visualizing one of the decision trees

        Parameters:
        ----------
        stage: [type]
            Stage to get tree from
        class_k: [type]
            tree for class class_k
        feature_names: [type]
            Feature names

        Returns:
        -------
        graphviz.Digraph:
            dot for tree diagram visual
        """
        return self.trees[stage][class_k].render(feature_names)
