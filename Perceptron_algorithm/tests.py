from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

X, y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

blobs_perceptron = Perceptron()
blobs_perceptron.fit(X_train, y_train)
predictions = blobs_perceptron.predict(X_test)

print(blobs_perceptron.assess_model(predictions, y_test))