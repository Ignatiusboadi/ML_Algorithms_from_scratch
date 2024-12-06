import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors (KNN) implementation for classification and regression tasks.

    Attributes:
        k (int): The number of neighbors to consider for predictions.
        task (str): The type of task - 'classification' or 'regression'.
        X (np.ndarray): The training feature matrix (set after calling `fit`).
        y (np.ndarray): The training labels/targets (set after calling `fit`).

    Methods:
        fit(X, y): Stores the training data.
        euclidean_distance(a, b): Computes the Euclidean distance between a point and a set of points.
        predict_single(x_i): Predicts the target for a single input sample.
        predict(X): Predicts the targets for multiple input samples.
    """

    def __init__(self, k=3, task='classification', distance_measure='euclidean'):
        """
        Initializes the KNN instance.

        Parameters:
            k (int): The number of neighbors to consider for predictions. Must be greater than 0.
            task (str): The type of task - either 'classification' or 'regression'.
            distance_measure: The type of measure to use for computing distances between points - euclidean, manhattan, hassanat.

        Raises:
            AssertionError: If `k` is not a positive integer.
            AssertionError: If `task` is not 'classification' or 'regression'.
            AssertionError: if `distance_measure` is not euclidean or manhattan or hassanat.
        """
        assert isinstance(k, int) and k > 0, 'k should be an integer and greater than 0.'
        assert task in ['classification', 'regression'], "task should either be 'classification' or 'regression'."
        assert distance_measure in ['euclidean', 'manhattan', 'hassanat'], "distance_measure should be one of euclidean, manhattan or hassanat."

        self.k = k
        self.task = task
        distance_measures = {'euclidean': self.euclidean_distance, 'manhattan': self.manhattan_distance, 'hassanat': self.hassanat_distance}
        self.distance_measure = distance_measures[distance_measure]

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
        assert self.k < X.shape[0], 'k should be less than the number of points in X.'

        self.X = X
        self.y = y

    def euclidean_distance(self, a, b):
        """
        Computes the Euclidean distance between a single point and a set of points.

        Parameters:
            a (np.ndarray): A 1D array representing the single input point.
            b (np.ndarray): A 2D array where each row is a point in the training set.

        Returns:
            np.ndarray: A 1D array of distances between `a` and each point in `b`.

        Raises:
            AssertionError: If `a` or `b` is not a numpy array.
            AssertionError: If `b` is not a 2D array.
        """
        assert isinstance(a, np.ndarray), 'a should be a numpy array.'
        assert isinstance(b, np.ndarray), 'b should be a numpy array.'
        assert b.ndim == 2, 'b should be a 2-dimensional array.'

        distance = ((a - b) ** 2).sum(axis=-1)

        return np.sqrt(distance)
    
    def manhattan_distance(self, a, b):
        """
        Computes the Manhattan distance between a single point and a set of points.

        Parameters:
            a (np.ndarray): A 1D array representing the single input point.
            b (np.ndarray): A 2D array where each row is a point in the training set.

        Returns:
            np.ndarray: A 1D array of distances between `a` and each point in `b`.

        Raises:
            AssertionError: If `a` or `b` is not a numpy array.
            AssertionError: If `b` is not a 2D array.
        """
        assert isinstance(a, np.ndarray), 'a should be a numpy array.'
        assert isinstance(b, np.ndarray), 'b should be a numpy array.'
        assert b.ndim == 2, 'b should be a 2-dimensional array.'
        distance = (np.abs(a - b)).sum(axis=-1)

        return distance
    
    def hassanat_distance(self, a, b):
        """
        Computes the Hassanat distance between a single point and a set of points. 
        An implementation from the paper `On Enhancing The Performance Of Nearest Neighbour Classifiers Using Hassanat Distance Metric`

        Parameters:
            a (np.ndarray): A 1D array representing the single input point.
            b (np.ndarray): A 2D array where each row is a point in the training set.

        Returns:
            np.ndarray: A 1D array of distances between `a` and each point in `b`.

        Raises:
            AssertionError: If `a` or `b` is not a numpy array.
            AssertionError: If `b` is not a 2D array.
        """

        assert isinstance(a, np.ndarray), 'a should be a numpy array.'
        assert isinstance(b, np.ndarray), 'b should be a numpy array.'
        assert b.ndim == 2, 'b should be a 2-dimensional array.'

        distance = np.where(np.minimum(a, b) >= 0, 1 - (1 + np.minimum(a, b)) / (1 + np.maximum(a, b)), 1 - (1 + np.minimum(a, b) + np.abs(np.minimum(a, b)))/(1 + np.maximum(a, b) + np.abs(np.minimum(a, b))))

        return distance.sum(axis=-1)

    def predict_single(self, x_i):
        """
        Predicts the target for a single input sample.

        Parameters:
            x_i (np.ndarray): A 1D array representing the input sample.

        Returns:
            int or float: The predicted label (for classification) or value (for regression).

        Raises:
            AssertionError: If `x_i` is not a numpy array.
        """
        assert hasattr(self, 'X') and hasattr(self, 'y'), "The model has not been fitted yet."
        assert type(x_i) == np.ndarray, 'x_i must be a numpy array.'


        distances = self.distance_measure(x_i, self.X)
        sorted_distances = np.argsort(distances)
        topk_distances = sorted_distances[:self.k]
        topk_labels = self.y[topk_distances]

        if self.task == 'classification':
            prediction = Counter(topk_labels).most_common(1)[0][0]
        elif self.task == 'regression':
            prediction = np.mean(topk_labels)

        return prediction

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

        distances = self.distance_measure(X[:, np.newaxis], self.X) # np.sqrt(((X[:, np.newaxis] - self.X) ** 2).sum(axis=2))

        topk_indices = np.argsort(distances, axis=1)[:, :self.k]
        topk_labels = self.y[topk_indices]

        if self.task == 'classification':
            predictions = np.array([Counter(labels).most_common(1)[0][0] for labels in topk_labels])
        elif self.task == 'regression':
            predictions = np.mean(topk_labels, axis=1)

        return predictions
    
    def assess_model(self, predictions, actual):
        """
        Calculates the accuracy of the predictions compared to the actual values.

        Parameters:
            predictions (np.ndarray): A 1D array of predicted labels or values.
            actual (np.ndarray): A 1D array of true labels or values.

        Returns:
            str: The accuracy as a percentage formatted to two decimal places (e.g., "95.67").

        Raises:
            AssertionError: If `predictions` and `actual` do not have the same length.
        """
        assert predictions.shape[0] == actual.shape[0], 'predictions and actual vectors should have the same lengths.'

        if self.task == 'classification':
            return f"Accuracy: {np.mean(predictions == actual) * 100:.2f}"
        else:
            r2_score = np.sum((predictions - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            return f"R2 Score: {r2_score:.4f}"
    