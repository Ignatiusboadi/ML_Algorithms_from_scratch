import numpy as np
from knn import KNN

class INN(KNN):
    def __init__(self, task='classification', distance_measure='euclidean'):
        self.task = task
        self.distance_measure = distance_measure

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        pass