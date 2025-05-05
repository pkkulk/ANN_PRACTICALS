import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 100)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function (for multiple values)
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

# Tanh function
def tanh(x):
    return np.tanh(x)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Plotting all functions
plt.figure(figsize=(20, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Function")
plt.grid(True)

# Softmax (only for a fixed vector, not full range)
plt.subplot(2, 2, 2)
input_vals = np.array([1, 2, 3, 4, 5])
plt.plot(input_vals, softmax(input_vals))
plt.title("Softmax Function")
plt.grid(True)

# Tanh
plt.subplot(2, 2, 3)
plt.plot(x, tanh(x))
plt.title("Tanh Function")
plt.grid(True)

# ReLU
plt.subplot(2, 2, 4)
plt.plot(x, relu(x))
plt.title("ReLU Function")
plt.grid(True)

plt.tight_layout()
plt.show()
