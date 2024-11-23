import numpy as np

class NaiveBayes:
    """
    Naive Bayes classifier implementation for continuous features using Gaussian likelihood.

    Attributes:
        classes (np.ndarray): Unique class labels found in the training data.
        mean (np.ndarray): Mean of features for each class, computed during training.
        var (np.ndarray): Variance of features for each class, computed during training.
        priors (np.ndarray): Prior probabilities for each class, computed during training.

    Methods:
        fit(X, y): Fits the model by calculating class-specific means, variances, and priors.
        predict(X): Predicts class labels for a batch of input samples.
        _predict(x_i): Predicts the class label for a single input sample.
        _pdf(class_idx, x_i): Computes the Gaussian PDF for a given class and input sample.
        check_accuracy(actual, predictions): Computes the accuracy of predictions compared to actual labels.
    """

    def fit(self, X, y):
        """
        Fits the Naive Bayes model to the training data.

        Parameters:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).

        Raises:
            AssertionError: If `X` is not a 2D numpy array or `y` is not a numpy array.
        """
        assert type(X) == np.ndarray, 'X should be a numpy array.'
        assert X.ndim == 2, 'X should be 2-dimensional.'
        assert type(y) == np.ndarray, 'y should be a numpy array.'
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[c == y]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predicts class labels for a batch of input samples.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            list: Predicted class labels for each input sample.

        Raises:
            AssertionError: If `X` is not a 2D numpy array.
        """
        assert hasattr(self, 'X') and hasattr(self, 'y'), "The model has not been fitted yet."
        assert type(X) == np.ndarray, 'X should be a numpy array.'
        assert X.ndim == 2, 'X should be 2-dimensional.'
        predictions = [self._predict(x_i) for x_i in X]
        return predictions

    def _predict(self, x_i):
        """
        Predicts the class label for a single input sample.

        Parameters:
            x_i (np.ndarray): Input feature array of shape (n_features,).

        Returns:
            int or str: Predicted class label.
        """
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x_i)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x_i):
        """
        Computes the Gaussian probability density function for a given class and input sample.

        Parameters:
            class_idx (int): Index of the class.
            x_i (np.ndarray): Input feature array of shape (n_features,).

        Returns:
            np.ndarray: Probabilities for each feature given the class.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-9
        p_x_i = (np.exp(- (x_i - mean) ** 2 / (2 * var))) / (np.sqrt(2 * np.pi * var))
        return p_x_i
        
    def check_accuracy(self, actual, predictions):
        """
        Computes the accuracy of predictions compared to the actual labels.

        Parameters:
            actual (np.ndarray): True labels of shape (n_samples,).
            predictions (np.ndarray): Predicted labels of shape (n_samples,).

        Returns:
            str: Formatted string showing accuracy as a percentage.

        Raises:
            AssertionError: If `actual` and `predictions` have different lengths.
        """
        assert actual.shape[0] == predictions.shape[0], 'Predictions and actuals should be of the same length.'
        accuracy = np.mean(actual == predictions) * 100

        return f'Accuracy: {accuracy:.2f}%'
