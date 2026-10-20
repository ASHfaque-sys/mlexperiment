from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target

model = SVC(kernel='linear')
model.fit(X, y)

print("Prediction:", model.predict([X[0]]))
print("Accuracy:", model.score(X, y))