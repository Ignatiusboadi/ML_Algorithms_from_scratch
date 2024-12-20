from sklearn import preprocessing, datasets
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayes

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

iris_nb = NaiveBayes()
iris_nb.fit(X_train, y_train)

predictions = iris_nb.predict(X_test)

print('Naive Bayes classification accuracy', iris_nb.check_accuracy(y_test, predictions))
