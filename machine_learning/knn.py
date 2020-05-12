import numpy as np


class Knn():

    def __init__(self, k=3, save_history=False, tolerance=0.001):
        """knn model

        Parameters:
        ----------
        k: int, optional, default 3
            number of clusters
        save_history: bool, optional, default False
            Whether to save intermediate steps, for analysis and 
            visualisation - see notebook for example
        tolerance: float, optional, default 0.001
            Stopping tolerance for change in centorids
        """
        self.k = k
        self.save_history = save_history
        self.centroids = [None for _ in range(k)]
        self.iteration = 0
        self.tolerance = tolerance

    def update_centroids(self, X):
        """Update the centroids as the mean of clusters

        The new centroids are calculated as the mean of the clusters
        based on the current cluster assignment (self.cluster_labels).
        After the update the new assignment is done to update the
        labels

        Parameters:
        ----------
        X: numpy.ndarray
            Input data to cluster, shape (m samples, n features)

        Returns:
        -------
        float:
            Sum of the euclidean distance change in each of the 
            centroids after updating
        """
        new_centroids = []
        for i in range(self.k):
            new_centroid = X[self.cluster_labels == i, :].mean(axis=0)
            new_centroids.append(new_centroid)
        new_centroids = np.stack(new_centroids, axis=0)
        self.iteration += 1
        distance_change = self.dist(self.centroids, new_centroids).sum()
        self.centroids = new_centroids
        new_cluster_labels = self.assign_clusters(X)
        self.cluster_labels = new_cluster_labels
        if self.save_history:
            self.centroid_history.append(self.centroids)
            self.cluster_labels_history.append(self.cluster_labels)
        return distance_change

    def fit(self, X, max_updates=10):
        """Fit the knn model

        Fitting the model updates the centroids and cluster labels
        iteratively util the centroids no longer change or the max
        number of iterations is reached

        Parameters:
        ----------
        X: numpy.ndarray
            Input data to cluster, shape (m samples, n features)
        max_updates: int, optional, default 10
            Maximum number of iterations permitted
        """
        self.initalise_centroids(X)
        distance_change = 10**6
        while self.iteration < max_updates and not distance_change < self.tolerance:
            distance_change = self.update_centroids(X)
        print(f'Finished at iteration {self.iteration}')

    def initalise_centroids(self, X):
        """Sets initial centorids randomly and assigns cluster labels

        The centroids are chosen randomly based on observed range of
        values in X

        Parameters:
        ----------
        X: numpy.ndarray
            Input data to cluster, shape (m samples, n features)
        """
        X_mins = X.min(axis=0)
        X_maxs = X.max(axis=0)
        self.centroids = np.stack(
            [np.random.uniform(xi_min, xi_max, self.k)
             for xi_min, xi_max in zip(X_mins, X_maxs)],
            axis=-1)
        self.cluster_labels = self.assign_clusters(X)
        if self.save_history:
            self.centroid_history = [self.centroids]
            self.cluster_labels_history = [self.cluster_labels]

    def dist(self, a, b, axis=1):
        """Euclidean distance function

        Parameters:
        ----------
        a: numpy.ndarray
            samples, shape (m sample, n features)
        b: numpy.ndarray
            centroid, shape (n_features,)
        axis: int, optional, default 1
            Set to 1 to sum along rows

        Returns:
        -------
        numpy.ndarray:
            Distance between each sample and centroid
        """
        return np.linalg.norm(a - b, axis=axis)

    def assign_clusters(self, X):
        """Assigns each sample of X to its nearest cluster centroid

        Parameters:
        ----------
        X: numpy.ndarray
            Input data to cluster, shape (m samples, n features)

        Returns:
        -------
        numpy.ndarray:
            Cluster label for each sample, shape (m samples, 1)
        """
        distances = []
        for centroid in self.centroids:
            centorid_distances = self.dist(X, centroid)
            distances.append(centorid_distances)
        all_distaces = np.stack(distances, axis=1)
        cluster_labels = np.argmin(all_distaces, axis=1)
        return cluster_labels
