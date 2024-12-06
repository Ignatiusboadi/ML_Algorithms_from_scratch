import numpy as np
from knn import KNN

class IINC(KNN):
    """
    Inverse-distance-weighted k-Nearest Neighbors (IINC) model for classification or regression.

    This model extends k-Nearest Neighbors by using inverse-distance weighting to calculate
    probabilities or outputs for predictions.
    """

    def __init__(self, task='classification', distance_measure='euclidean'):
        """
        Initializes the IINC model.

        Parameters:
            task (str): Task type, either 'classification' or 'regression'. Defaults to 'classification'.
            distance_measure (str): The distance metric to use (e.g., 'euclidean'). Defaults to 'euclidean'.

        Raises:
            KeyError: If the specified distance_measure is not available in the parent KNN class.
        """
        self.task = task
        self.distance_measure = self.distance_measures[distance_measure].__get__(self)

    def fit(self, X, y):
        """
        Stores the training data for the IINC model.

        Parameters:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) representing the training feature matrix.
            y (np.ndarray): A 1D array of shape (n_samples,) representing the training labels or targets.

        Raises:
            AssertionError: If `X` is not a 2D array.
            AssertionError: If the number of rows in `X` and `y` do not match.
            AssertionError: If `y` labels are not consecutively encoded from 0 to max without gaps.
        """
        assert X.ndim == 2, 'X should be a 2-dimensional array.'
        assert y.shape[0] == X.shape[0], 'Number of rows in y must match the number of rows in X.'
        assert np.allclose(np.unique(y), np.arange(0, np.max(y) + 1)), \
            "Labels in y must be consecutively encoded from 0 to max without gaps."

        self.X = X
        self.y = y
        self.labels = np.unique(self.y)

    def predict(self, X):
        """
        Predicts the targets for multiple input samples using inverse-distance weighting.

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

        # Calculate distances from each input sample to all training samples.
        distances = self.distance_measure(X[:, np.newaxis], self.X)
        
        # Sort distances to find the nearest neighbors.
        sorted_distances = np.argsort(distances, axis=1)
        argsorted_y = self.y[sorted_distances]

        # Calculate weights as inverse positions of sorted distances.
        label_positions = np.arange(1, self.X.shape[0] + 1)  # Positions start from 1.
        label_probabilities = np.zeros((X.shape[0], self.labels.size))

        # Compute weighted probabilities for each label.
        for i, label in enumerate(self.labels):
            matches = argsorted_y == label
            inverse_positions = 1 / label_positions[np.newaxis, :]  # Weighting by inverse distance.
            label_probabilities[:, i] = (matches * inverse_positions).sum(axis=1)

        # Return the label with the highest probability for classification.
        return np.argmax(label_probabilities, axis=1)
