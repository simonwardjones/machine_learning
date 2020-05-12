import itertools
import logging

import numpy as np
from graphviz import Digraph

logging.basicConfig()
logger = logging.getLogger('decision_tree')
logger.setLevel(logging.INFO)


class TreeNode():

    count = itertools.count()

    def __init__(self,
                 data,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 n_classes=2,
                 max_features=None,
                 depth=0,
                 impurity='gini',
                 is_classifier=True):
        """
        A single node in a decision tree

        After recursive splitting of the input data, a given node 
        represents one split of the tree if it is not a leaf node. The
        leaf node stores the training samples in that leaf to be used 
        for prediction. 
        The splitting nodes record the feature to split on as attribute 
        self.best_feature_index and the splitting value as attribute
        self.best_feature_split_val

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features + 1 target)
            Note the last column of the data are the target values
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
            used to split at the node, the number of features used in
            this case is sqrt(n_features).
            Else all the features are considered when splitting at this
            node
        depth: int, optional, default 0
            The depth of the node in the tree
        impurity: str, optional, default 'gini'
            The impurity measure to use when splitting at the node.
            I have currently only implemented two
            'gini' - Uses the gini impurity (for classification)
            'mse' - Uses the mean square error - equal to variance (for
            regression)
        is_classifier: bool, optional, default True
            Is the tree node used as part of a classification problem
            or a regression problem. Should be set to True if
            classification, False if regression
        """
        self.data = data
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.max_features = max_features
        self.depth = depth
        self.impurity = impurity
        self.is_classifier = is_classifier

        self.data_shape = data.shape
        self.split_attempted = False
        self.best_split_impurity = None
        self.best_feature_index = None
        self.best_feature_split_val = None
        self.is_leaf = False
        self.node_impurity = self.calculate_impurity([data[:, -1]])
        self.value = self._init_value(data)
        self.id = str(next(self.count))

    def __repr__(self):
        return (
            f'<TreeNode '
            f'depth:{self.depth} '
            f'node_impurity:{self.node_impurity:.2f} '
            f'samples:{self.data_shape[0]} '
            f'{"🌳" if self.is_root else ""}'
            f'{"🍁" if self.is_leaf else ""}'
            f'>')

    @property
    def is_root(self):
        return self.depth == 0

    def info(self):
        return dict(
            data_shape=self.data_shape,
            n_classes=self.n_classes,
            depth=self.depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            node_impurity=self.node_impurity,
            split_attempted=self.split_attempted,
            best_split_impurity=self.best_split_impurity,
            best_feature_index=self.best_feature_index,
            best_feature_split_val=self.best_feature_split_val,
            is_root=self.is_root)

    def _init_value(self, data):
        """  
        Returns the terminal node value based on the input data

        For a classifier this is the class_counts.
        For a regressor this is the average y value. 

        Note this value can be access at a splitting node to see what
        the prediction would have been at that level of the tree

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features + 1 target)
            Note the last column of the data are the target values

        Returns:
        -------
        numpy.ndarray or float:
            Class counts if classifier, else mean of target values 
        """
        if self.is_classifier:
            return np.bincount(
                data[:, -1].astype(int),
                minlength=self.n_classes)
        else:
            return np.mean(data[:, -1])

    def split(self, feature_index, feature_split_val, only_y=True):
        """  
        Splits self.data on feature with index feature_index using
        feature_split_val.

        Each sample is included in left output if the feature value for
        the sample is less than or equal to the feature_split_val else 
        it is included in the right output

        Parameters:
        ----------
        feature_index: int
            Index of the feature (column) in self.data
        feature_split_val: float
            Feature value to use when splitting data
        only_y: bool, optional, default True
            Return only the y values in left and right - this is used 
            when checking candidate split purity increase

        Returns:
        -------
        (numpy.ndarray, numpy.ndarray):
            left and right splits of self.data
        """
        assert feature_index in range(self.data.shape[1])
        if only_y:
            select = -1
        else:
            select = slice(None)
        left_mask = self.data[:, feature_index] <= feature_split_val
        right_mask = ~ left_mask
        left = self.data[left_mask, select]
        right = self.data[right_mask, select]
        logger.debug(
            f'Splitting on feature_index {feature_index} with '
            f'feature_split_val = {feature_split_val} creates left '
            f'with shape {left.shape} and right with '
            f'shape {right.shape}')
        return left, right

    def gini_impurity(self, groups):
        """  
        Calculate the Gini impurity for groups of values

        The impurity returned is the weighted average of the impurity
        of the groups.

        You can think of gini impurity as the probability of incorrectly
        predicting a random sample from a group if the prediction was
        made based purely on the distribution of class labels in the
        group


        Parameters:
        ----------
        groups: tuple
            The groups tuple is made up of arrays of values. It is 
            often called with groups = (left, right) to find the purity
            of the candidate split

        Returns:
        -------
        float:
            Gini impurity
        """
        gini = 0
        total_samples = sum(group.shape[0] for group in groups)
        for i, group in enumerate(groups):
            group = group.astype(int)
            class_counts = np.bincount(group, minlength=self.n_classes)
            group_size = class_counts.sum()
            class_probs = class_counts / group_size
            unique_classes = np.count_nonzero(class_counts)
            group_gini = (class_probs * (1 - class_probs)).sum()
            gini += group_gini * (group_size / total_samples)
            logger.debug(
                f'Group {i} has size {group.shape[0]} with '
                f'{unique_classes} unique classes '
                f'with Gini index {group_gini:.3}')
        return gini

    def mean_square_impurity(self, groups):
        """  
        Calculates the mean square error impurity

        The mse impurity is the weighted average of the group variances

        Parameters:
        ----------
        groups: tuple
            The groups tuple is made up of arrays of values. It is 
            often called with groups = (left, right) to find the purity
            of the candidate split

        Returns:
        -------
        float:
            Mean square error impurity
        """
        mean_square_error = 0
        total_samples = sum(group.shape[0] for group in groups)
        for i, group in enumerate(groups):
            group_size = group.shape[0]
            group_mean = np.mean(group)
            group_mean_square_error = np.mean((group - group_mean) ** 2)
            mean_square_error += group_mean_square_error * \
                (group_size / total_samples)
            logger.debug(
                f'Group {i} has size {group.shape[0]} with '
                f'with MSE impurity {group_mean_square_error:.3}')
        logger.debug(f'MSE candidate {mean_square_error}')
        return mean_square_error

    def calculate_impurity(self, groups):
        """  
        Calculates impurity based on self.impurity setting

        Parameters:
        ----------
        groups: tuple
            The groups tuple is made up of arrays of values. It is 
            often called with groups = (left, right) to find the purity
            of the candidate split

        Returns:
        -------
        float:
            Mean square error of groups if self.impurity = 'mse'
            Gini impurity of groups if self.impurity = 'mse'
        """
        if self.impurity == 'gini':
            return self.gini_impurity(groups)
        elif self.impurity == 'mse':
            return self.mean_square_impurity(groups)

    def check_split(self, feature_index, feature_split_val):
        """  
        Updates best split if candidate split is better

        Splits the data in groups using self.split. Checks min samples
        leaf condition after split. Calculates impurity of the split
        then if impurity is less than best split already found and less
        than the current node impurity the best_feature_index, the 
        best_feature_split_val and the best_split_impurity values are
        updated.

        Parameters:
        ----------
        feature_index: int
            Index of the feature (column) in self.data
        feature_split_val: float
            Feature value to use when splitting data
        """
        groups = self.split(feature_index, feature_split_val)
        if any(len(group) < self.min_samples_leaf for group in groups):
            logger.debug(
                f"Can't split node on feature {feature_index} with split "
                f"val {feature_split_val} due to min_samples_leaf condition")
            return None
        split_impurity = self.calculate_impurity(groups)
        best_current_impurity = (
            10**10 if self.best_split_impurity is None
            else self.best_split_impurity)
        if ((split_impurity < best_current_impurity) and
                (split_impurity < self.node_impurity)):
            logger.debug(
                f'Found new best split with feature_split_val='
                f'{feature_split_val} for feature_index = {feature_index} '
                f'and split_impurity = {split_impurity:.2f}')
            self.best_feature_index = feature_index
            self.best_feature_split_val = feature_split_val
            self.best_split_impurity = split_impurity

    def find_best_split(self):
        """
        Finds best split at the node

        Loops through each feature and each unique value of that feature
        checking for the best candidate split (i.e. the split that 
        reduces the impurity the most)

        The function first checks if we have reached the max depth or if
        self.data < self.min_samples_split. In either case no further
        split is allowed and the function returns

        All features are considered unless self.max_features == 'sqrt'
        in which case a random subset of features are used of size
        sqrt(n_features)
        """
        if self.depth == self.max_depth:
            return
        if self.data.shape[0] < self.min_samples_split:
            logger.info(f"{self} can't split as samples < min_samples_split")
            return None
        if self.node_impurity == 0:
            logger.info(f"Can't improve as node pure")
            return None
        n_features = self.data.shape[1] - 1
        all_feature_indices = np.arange(n_features)
        if self.max_features == 'sqrt':
            features_to_check = np.random.choice(
                all_feature_indices,
                size=np.sqrt(n_features).astype(int))
        else:
            features_to_check = all_feature_indices
        logger.info(f'Checking features {features_to_check}')
        for feature_index in features_to_check:
            for feature_split_val in np.unique(self.data[:, feature_index]):
                self.check_split(feature_index, feature_split_val)
        self.split_attempted = True

    def recursive_split(self):
        """  
        Recursively grows tree by splitting to reduce impurity the most

        The function finds the best split using the find_best_split
        method. If there was a split found two nodes are created - left
        and right. Finally the recursive_split method is called on each
        of the new nodes.

        Note the depth of the children node is incremented, otherwise
        the node settings such as min_samples_split are passed to the
        children nodes
        """
        self.find_best_split()
        if self.best_feature_index is not None:
            logger.info(f'Splitting tree on feature_index '
                        f'{self.best_feature_index} and feature_split_val '
                        f'{self.best_feature_split_val:.2f}')
            left, right = self.split(
                feature_index=self.best_feature_index,
                feature_split_val=self.best_feature_split_val,
                only_y=False)
            del self.data
            self.left = TreeNode(
                data=left,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_classes=self.n_classes,
                max_features=self.max_features,
                depth=self.depth + 1,
                impurity=self.impurity,
                is_classifier=self.is_classifier)
            self.right = TreeNode(
                data=right,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_classes=self.n_classes,
                max_features=self.max_features,
                depth=self.depth + 1,
                impurity=self.impurity,
                is_classifier=self.is_classifier)
            self.left.recursive_split()
            self.right.recursive_split()
        else:
            logger.info('Reached max depth or no splits reduce impurity')
            self.is_leaf = True

    def walk_depth_first(self, only_leaves=True):
        """  
        Generator traversing of all nodes below and including this node

        Depth first so visiting children before siblings

        Parameters:
        ----------
        only_leaves: bool, optional, default True
            Only return leaf nodes

        Yields:
            TreeNode: each node in tree
        """
        if self.is_leaf:
            yield self
        else:
            if not only_leaves:
                yield self
            for node in (self.left, self.right):
                yield from node.walk_depth_first(only_leaves)

    def walk_breadth_first(self, layer=None):
        """  
        Generator traversing of all nodes below and including this node

        Breadth first so visiting siblings before children

        Parameters:
        ----------
        only_leaves: bool, optional, default True
            Only return leaf nodes

        Yields:
            TreeNode: each node in tree
        """
        if layer is None:
            layer = [self]
        for node in layer:
            yield node
        new_layer = [
            child
            for node_children in [[node.left, node.right]
                                  for node in layer if not node.is_leaf]
            for child in node_children]
        if new_layer:
            yield from self.walk_breadth_first(new_layer)

    def print_tree(self):
        """  
        prints ascii representation of tree below this node
        """
        for node in self.walk_depth_first(only_leaves=False):
            print('--' * node.depth + str(node))

    def predict_row_proba(self, row):
        """
        Predicts class probabilities for input row by walking the tree
        and returning the leaf node class probabilities

        Parameters:
        ----------
        row: numpy.ndarray
            Input row, shape (n features,)

        Returns:
        -------
        numpy.ndarray:
            Class probabilities, shape (n classes, )
        """
        if self.is_leaf:
            group_size = self.value.sum()
            class_probs = self.value / group_size
            return class_probs
        elif row[self.best_feature_index] <= self.best_feature_split_val:
            return self.left.predict_row_proba(row)
        else:
            return self.right.predict_row_proba(row)

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
        if not self.is_classifier:
            raise Exception('Not a classifier')
        if len(data.shape) == 2:
            return np.stack([self.predict_row_proba(row)
                             for row in data])
        else:
            return self.predict_row_proba(data)

    def predict_regressor_row(self, row):
        """
        Predicts target value for input row by walking the tree
        and returning the leaf node value

        Parameters:
        ----------
        row: numpy.ndarray
            Input row, shape (n features,)

        Returns:
        -------
        float:
            Predicted target value
        """
        if self.is_leaf:
            return self.value
        elif row[self.best_feature_index] <= self.best_feature_split_val:
            return self.left.predict_regressor_row(row)
        else:
            return self.right.predict_regressor_row(row)

    def predict_regressor(self, data):
        """  
        Predicts target values for each row in data by walking the
        tree and returning the leaf node values

        Parameters:
        ----------
        data: numpy.ndarray
            The input data with shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Predicted target values, shape (m samples, 1)
        """
        if len(data.shape) == 2:
            return np.stack([self.predict_regressor_row(row)
                             for row in data])
        else:
            return self.predict_regressor_row(data)

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
        if self.is_classifier:
            return np.argmax(self.predict_proba(data), axis=-1)
        else:
            return self.predict_regressor(data)

    def dot(self,
            feature_names,
            samples=True,
            impurity=True,
            value=True):
        """  
        Returns Digraph visualizing the tree below this node

        Parameters:
        ----------
        feature_names: list[str]
            List of feature names
        samples: bool, optional, default True
            Whether to display the number of samples on this node
        impurity: bool, optional, default True
            Whether to display the impurity value on this node
        value: bool, optional, default True
            Whether to dispaly the value on this node

        Returns:
        -------
        graphviz.Digraph:
            dot for tree diagram visual
        """
        dot = Digraph(
            comment='Decsion Tree',
            node_attr=dict(shape="rectangle",
                           style="rounded",
                           fillcolor="#028d35"))
        for i, node in enumerate(self.walk_breadth_first()):
            label = ""
            if not node.is_leaf:
                label += (
                    f'{feature_names[node.best_feature_index]} <= '
                    f'{node.best_feature_split_val}\n')
                dot.edge(node.id, node.left.id)
                dot.edge(node.id, node.right.id)
            if samples:
                label += f'Samples = {node.data_shape[0]}\n'
            if impurity:
                label += f'Impurity = {node.node_impurity:.2f}\n'
            if value:
                if self.is_classifier:
                    label += f'Class counts = {str(node.value)}\n'
                else:
                    label += f'Average y = {node.value:.2f}\n'
            dot.node(name=node.id, label=label)
        return dot
