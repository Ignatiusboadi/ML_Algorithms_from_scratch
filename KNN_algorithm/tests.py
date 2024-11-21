from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from knn import KNN
clf = KNN(k=5, task='classification')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(clf.check_accuracy(predictions, y_test))

print((predictions == y_test).sum())