import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR problem)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output for XOR
y = np.array([[0], [1], [1], [0]])

# Set seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))

# Biases
b1 = np.random.uniform(size=(1, hidden_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Learning rate and epochs
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # FORWARD PASS
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)

    # BACKPROPAGATION
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final prediction
print("\nFinal Output after training:")
print(output)
