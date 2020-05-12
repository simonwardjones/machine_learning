import logging

import numpy as np

from .tree import TreeNode

logging.basicConfig()
logger = logging.getLogger('decision_tree')
logger.setLevel(logging.INFO)


class DecisionTree():

    def __init__(self,
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_classes=2,
                 max_features=None,
                 impurity='gini',
                 is_classifier=True):
        """Decision tree model

        Parameters:
        ----------
        max_depth: int
            The maximum depth allowed when "growing" a tree
        min_samples_split: int
            The minimum number of samples required to allow a split at
            a the node
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
        impurity: str, optional, default 'gini'
            The impurity measure to use when splitting at each node.
            I have currently only implemented two
            'gini' - Uses the gini impurity (for classification)
            'mse' - Uses the mean square error - equal to variance (for
            regression)
        is_classifier: bool, optional, default True
            Is the model used as part of a classification problem
            or a regression problem. Should be set to True if
            classification, False if regression
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.max_features = max_features
        self.impurity = impurity
        self.is_classifier = is_classifier

        self.is_fitted = False
        self.tree = None

    def fit(self, X, y):
        """Fits the decision tree model

        The tree is fitted by instantiaing a root TreeNode instance and
        then calling the recursive_split method. This iteratively grows
        the tree by finding the best split to reduce the impurity the
        most.

        Parameters:
        ----------
        X: numpy.ndarray
            Training data, shape (m samples, n features)
        y: numpy.ndarray
            Target values, shape (m samples, 1)
            If classifier with n_classes the values are assumed to be in
            0, ..., n-1
        """
        y_shape = (X.shape[0], 1)
        data = np.concatenate((X, y.reshape(y_shape)), axis=1)
        self.tree = TreeNode(
            data=data,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_classes=self.n_classes,
            max_features=self.max_features,
            impurity=self.impurity,
            is_classifier=self.is_classifier)
        self.tree.recursive_split()
        self.is_fitted = True

    def predict(self, data):
        """Predicts target values or class labels for classification

        Predicts target values/class for each row in data by walking the
        tree and returning the leaf node value for regression or the 
        class with the largest predicted probability for classification

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted target values or class labels for classification
        """
        if not self.is_fitted:
            raise Exception('Decision tree not fitted')
        return self.tree.predict(data)

    def predict_proba(self, data):
        """Predicts class probabilities for input data

        Predicts class probabilities for each row in data by walking the
        tree and returning the leaf node class probabilities

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted sample class probabilities, 
            shape (m samples, n classes)
        """
        if not self.is_fitted:
            raise Exception('Decision tree not fitted')
        return self.tree.predict_proba(data)

    def render(self, feature_names):
        """Returns Digraph visualizing the decision tree (if fitted)

        Parameters:
        ----------
        feature_names: list[str]
            List of feature names

        Returns:
        -------
        graphviz.Digraph:
            dot for tree diagram visual
        """
        if not self.is_fitted:
            print('Decision tree not fitted')
        else:
            return self.tree.dot(feature_names=feature_names)
