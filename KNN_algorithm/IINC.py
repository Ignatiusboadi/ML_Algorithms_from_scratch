import numpy as np
from knn import KNN

class IINC(KNN):
    def __init__(self, task='classification', distance_measure='euclidean'):
        self.task = task
        self.distance_measure = self.distance_measures[distance_measure].__get__(self)

    def fit(self, X, y):
        """
        Stores the training data.

        Parameters:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) representing the training feature matrix.
            y (np.ndarray): A 1D array of shape (n_samples,) representing the training labels or targets.

        Raises:
            AssertionError: If `X` is not a 2D array.
            AssertionError: If the number of rows in `X` and `y` do not match.
            AssertionError: If the labels are not encoded from 0 to max without gaps.
        """
        assert X.ndim == 2, 'X should be a 2-dimensional array.'
        assert y.shape[0] == X.shape[0], 'number of rows in targets/labels vector,y should match number of rows in feature matrix, X.'
        assert np.allclose(np.unique(y), np.arange(0, np.max(y) + 1)), "Labels must be consecutively encoded from 0 to max without gaps."

        self.X = X
        self.y = y
        self.labels = np.unique(self.y)

    def predict(self, X):
        """
        Predicts the targets for multiple input samples.

        Parameters:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) representing the input feature matrix.

        Returns:
            np.ndarray: A 1D array of predictions for each input sample.

        Raises:
            AssertionError: If the model has not been fitted.
            AssertionError: If `X` is not a 2D array.
        """
        assert hasattr(self, 'X') and hasattr(self, 'y'), "The model has not been fitted yet."
        assert X.ndim == 2, 'X should be a 2-dimensional array.'

        distances = self.distance_measure(X[:, np.newaxis], self.X)
        sorted_distances = np.argsort(distances, axis=1)
        argsorted_y = self.y[sorted_distances]

        label_positions = np.arange(1, self.X.shape[0] + 1)
        label_probabilities = np.zeros((X.shape[0], self.labels.size))

        for i, label in enumerate(self.labels):
            matches = argsorted_y == label
            inverse_positions = 1 / label_positions[np.newaxis, :]
            label_probabilities[:, i] = (matches * inverse_positions).sum(axis=1)

        return np.argmax(label_probabilities, axis=1)
