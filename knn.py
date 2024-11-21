import numpy as np
from collections import Counter


class KNN:
    '''

    '''
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_single(self, x_i):
        distances = np.array([np.linalg.norm(x_i, x_j) for x_j in self.X])
        sorted_distances = np.argsort(distances)
        topk_distances = sorted_distances[:self.k]
        topk_labels = self.y[topk_distances]
        predicted_label = Counter(topk_labels).most_common(1)[0][0]
        return predicted_label

    def predict(self, X):
        predictions = np.array([self.predict_single(x_i) for x_i in self.X])
        return predictions
    