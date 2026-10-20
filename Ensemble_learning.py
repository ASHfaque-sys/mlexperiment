from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

iris = load_iris()
X, y = iris.data, iris.target

bag = RandomForestClassifier(n_estimators=10)
boost = AdaBoostClassifier(n_estimators=10)

bag.fit(X, y)
boost.fit(X, y)

print("Bagging Accuracy:", bag.score(X, y))
print("Boosting Accuracy:", boost.score(X, y))