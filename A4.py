import numpy as np
import matplotlib.pyplot as plt

# Sample 2D data (X1, X2) and labels (0 or 1)
X = np.array([
    [1, 1],
    [2, 1],
    [2, 3],
    [3, 5],  # class 0 (blue)
    [6, 2],
    [7, 3],
    [8, 2],
    [9, 4]   # class 1 (red)
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
lr = 0.1

# Train the perceptron
for epoch in range(10):
    print(len(X))
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        print("z=>",z)
        pred = 1 if z >= 0 else 0
        error = y[i] - pred
        print("error=>",error)
        w += lr * error * X[i]
        print("W=>",w)
        b += lr * error

# Final learned values
print("Weights:", w)
print("Bias:", b)

# Plot the decision boundary
x_vals = np.linspace(0, 10, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
print("x-axis", x_vals ,"y-axis", y_vals)
plt.plot(x_vals, y_vals, '--k', label='Decision Boundary')

# Plot the points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 1' if i == 4 else "")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Boundary Example")
plt.grid(True)
plt.legend()
plt.show()
