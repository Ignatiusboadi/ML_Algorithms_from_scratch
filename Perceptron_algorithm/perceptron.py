import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def activation_func(self, x):
        return np.where(x >=0, 1, 0)
    

    def fit(self, X, y):
        self.X = X
        self.y = y
    