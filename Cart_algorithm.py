from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

print("Prediction:", model.predict([X[0]]))
print("Accuracy:", model.score(X, y))

plot_tree(model, filled=True)
plt.show()