import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        assert type(X) == np.ndarray, 'X should be a numpy array.'
        assert X.ndim == 2, 'X should be 2-dimensional.'
        assert type(y) == np.ndarray, 'y should be a numpy array.'
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self.mean[c, :] = X_c.mean(axis=0)
            self.var[c, :] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        predictions = [self._predict(x_i) for x_i in X]
        return predictions

    def _predict(self, x_i):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x_i)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x_i):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        p_x_i = (np.exp(- (x_i - mean) ** 2 / (2 * var))) / (np.sqrt(2 * np.pi * var))
        return p_x_i
        
    def check_accuracy(y_test, predictions):
        n_correct = sum(y_test == predictions) / y_test.shape[0]

        return f'Accuracy: {n_correct * 100:.2f}%'
