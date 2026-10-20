from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5]]
y = [20, 30, 40, 50, 60]

model = LinearRegression()
model.fit(X, y)

print("Prediction:", model.predict([[6]]))
print("R2 Score:", model.score(X, y))

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()