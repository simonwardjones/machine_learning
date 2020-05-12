import logging

import numpy as np

from .decision_tree import DecisionTree

logging.basicConfig()
logger = logging.getLogger('decision_tree')
logger.setLevel(logging.INFO)


class RandomForest():

    def __init__(self,
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_classes=2,
                 max_features='sqrt',
                 impurity='gini',
                 is_classifier=True,
                 n_trees=10,
                 bootstrap=True):
        """Random forest model

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
        n_trees: int, optional, default 10
            Number of trees in the forest
        bootstrap: bool, optional, default True
            Whether to bootstrap the data when fitting the trees
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.max_features = max_features
        self.impurity = impurity
        self.is_classifier = is_classifier

        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.is_fitted = False
        self.trees = []
        np.random.seed(1)

    def fit(self, X, y):
        """Fit the random forest model

        This method fits n_trees trees on the data with bootstrap
        samples. A random subset of the features is used at each split.


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
        for i, data in enumerate(self._samples(data)):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_classes=self.n_classes,
                max_features=self.max_features,
                impurity=self.impurity,
                is_classifier=self.is_classifier)
            logger.info(f'Fitting tree {i}')
            tree.fit(X, y)
            self.trees.append(tree)
        self.is_fitted = True

    def _samples(self, data):
        """Bootstrap sample generator

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features + 1 target)
            Note the last column of the data are the target values

        Yields:
            numpy.ndarray: Bootstrap sample of data
        """
        n_rows = data.shape[0]
        for _ in range(self.n_trees):
            if not self.bootstrap:
                yield data
            else:
                random_rows = np.random.choice(np.arange(n_rows),
                                               size=n_rows,
                                               replace=True)
                yield data[random_rows, :]

    def predict_proba(self, data):
        """Predicts class probabilities for input data

        The class probability predictions from each tree are averaged to
        provide the overall class prediction probabilities 

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
            raise Exception('Forest not fitted')
        # samples, classes, trees
        return np.stack(list(tree.predict_proba(data) for tree in self.trees),
                        axis=-1).sum(axis=-1) / self.n_trees

    def predict(self, data):
        """Predicts target values or class labels for classification

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted target values or class labels for classification
        """
        if self.is_classifier:
            return np.argmax(self.predict_proba(data), axis=-1)
        else:
            return np.stack(
                list(tree.predict(data) for tree in self.trees),
                axis=-1).mean(axis=-1)

    def render(self, tree_id, feature_names):
        """Returns Digraph visualizing one of the decision trees

        Parameters:
        ----------
        tree_id: [type]
            tree index to display
        feature_names: [type]
            Feature names

        Returns:
        -------
        graphviz.Digraph:
            dot for tree diagram visual
        """
        return self.trees[tree_id].render(feature_names)
