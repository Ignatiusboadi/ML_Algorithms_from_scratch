from sklearn.model_selection import train_test_split
from sklearn import datasets, preprocessing
from iinc import IINC
from knn import KNN


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

iris_clf = KNN(k=5, task='classification')
iris_clf.fit(X_train, y_train)
predictions = iris_clf.predict(X_test)
print('euclidean', iris_clf.assess_model(predictions, y_test))

iris_hassanat = KNN(k=5, task='classification', distance_measure='hassanat')
iris_hassanat.fit(X_train, y_train)
predictions = iris_hassanat.predict(X_test)
print('hassanant', iris_hassanat.assess_model(predictions, y_test))

iris_manhattan = KNN(k=5, task='classification', distance_measure='manhattan')
iris_manhattan.fit(X_train, y_train)
predictions = iris_manhattan.predict(X_test)
print('manhattan', iris_manhattan.assess_model(predictions, y_test))

iris_iinc = IINC(task='classification', distance_measure='manhattan')
iris_iinc.fit(X_train, y_train)
predictions = iris_iinc.predict(X_test)
print('manhattan_iinc', iris_iinc.assess_model(predictions, y_test))


housing = datasets.fetch_california_housing()
H_train, H_test, p_train, p_test = train_test_split(housing.data, housing.target, test_size=0.8, random_state=32)

scaler = preprocessing.StandardScaler()
H_train = scaler.fit_transform(H_train)
H_test = scaler.transform(H_test)

cali_reg = KNN(k=2, task='regression')
cali_reg.fit(H_train, p_train)
predictions = cali_reg.predict(H_test)
print(cali_reg.assess_model(predictions, p_test))
