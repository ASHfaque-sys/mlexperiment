from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.05)

model = DBSCAN(eps=0.3, min_samples=5)
labels = model.fit_predict(X)

print("Clusters:", set(labels))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()