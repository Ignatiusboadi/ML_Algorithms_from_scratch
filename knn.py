import numpy as np
from collections import Counter


class KNN:
    '''

    '''
    def __init__(self, k=3, task='classification'):
        assert type(k) == int and k > 0, 'k should be an integer and greater than 0.'
        assert task in ['classification', 'regression'], "task should either be 'classification' or 'regression'."
        self.k = k
        self.task = task

    def fit(self, X, y):
        
        self.X = X
        self.y = y

    def predict_single(self, x_i):
        distances = np.array([np.linalg.norm(x_i - x_j) for x_j in self.X])
        sorted_distances = np.argsort(distances)
        topk_distances = sorted_distances[:self.k]
        topk_labels = self.y[topk_distances]
        if self.task == 'classification':
            predicted_label = Counter(topk_labels).most_common(1)[0][0]
        elif self.task == 'regression':
            predicted_label = np.mean(topk_labels)

        return predicted_label

    def predict(self, X):
        predictions = np.array([self.predict_single(x_i) for x_i in self.X])
        return predictions
    