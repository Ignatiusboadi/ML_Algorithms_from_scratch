import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def activation_func(self, x):
        return np.where(x >= 0, 1, 0)
    

    def fit(self, X, y):
        self.X = X
        self.y = np.array([1 if y_i > 0 else 0 for y_i in y])
        self.bias = 0
        self.weights = np.zeros(self.X.shape[0])

    def train(self):
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                output = self.predict(x_i)
                update = self.lr * (self.y[idx] - output)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        dot_output = X @ self.weights + self.bias
        return self.activation_func(dot_output)
    