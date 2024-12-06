import numpy as np
from knn import KNN

class INN(KNN):
    def __init__(self, task='classification', distance_measure='euclidean'):
        self.task = task
        self.distance_measure = distance_measure

    def fit(self, X, y):
        """
        Stores the training data.

        Parameters:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) representing the training feature matrix.
            y (np.ndarray): A 1D array of shape (n_samples,) representing the training labels or targets.

        Raises:
            AssertionError: If `X` is not a 2D array.
            AssertionError: If the number of rows in `X` and `y` do not match.
        """
        assert X.ndim == 2, 'X should be a 2-dimensional array.'
        assert y.shape[0] == X.shape[0], 'number of rows in targets/labels vector,y should match number of rows in feature matrix, X.'

        self.X = X
        self.y = y

    def predict(self, X):
        pass
