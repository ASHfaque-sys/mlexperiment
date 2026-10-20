from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

print("PCA:", PCA(n_components=2).fit_transform(X).shape)
print("SVD:", TruncatedSVD(n_components=2).fit_transform(X).shape)
print("LDA:", LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y).shape)