import numpy as np

class Perceptron:
    """
    Implementation of a basic Perceptron for binary classification using a step activation function.

    Attributes:
        lr (float): Learning rate for weight updates.
        n_iters (int): Number of training iterations.
        weights (np.ndarray): Weight vector for features.
        bias (float): Bias term.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the Perceptron with the specified learning rate and number of iterations.

        Parameters:
            learning_rate (float): Step size for updating weights. Default is 0.01.
            n_iters (int): Maximum number of iterations for training. Default is 1000.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def activation_func(self, x):
        """
        Step activation function that maps input values to binary outputs.

        Parameters:
            x (float or np.ndarray): The input value(s).

        Returns:
            np.ndarray: Binary outputs (0 or 1).
        """
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        """
        Predicts the binary class labels for the input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features) or a single sample of shape (n_features,).

        Returns:
            np.ndarray: Predicted binary labels (0 or 1) for each sample.
        """
        dot_output = np.dot(X, self.weights) + self.bias  # Linear combination of weights and inputs
        return self.activation_func(dot_output)

    def fit(self, X, y):
        """
        Trains the Perceptron using the provided training data.

        Parameters:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) representing the training data.
            y (np.ndarray): A 1D array of shape (n_samples,) representing the binary target labels (0 or 1).

        Raises:
            AssertionError: If `X` is not a 2D array or `y` is not a 1D array.
            ValueError: If the number of samples in `X` and `y` do not match.
        """
        assert X.ndim == 2, 'X must be a 2-dimensional array.'
        assert y.ndim == 1, 'y must be a 1-dimensional array.'
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")
        
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError("Labels must be binary (0 and 1).")

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            converged = True
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.lr * (y[idx] - prediction)
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    converged = False
            if converged:
                break

    def assess_model(self, predictions, actuals):
        return f"Accuracy: {np.mean(predictions == actuals) * 100:.2f}"