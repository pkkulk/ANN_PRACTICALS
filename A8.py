import numpy as np

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # to avoid large numbers
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy Loss
def cross_entropy(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

# One Hot Encoding
def one_hot(y, num_classes):
    one_hot_labels = np.zeros((len(y), num_classes))
    one_hot_labels[np.arange(len(y)), y] = 1
    return one_hot_labels

# Sample Input (4 samples, 4 features each)
X = np.array([
    [0.1, 0.2, 0.7, 0.0],
    [0.5, 0.1, 0.0, 0.4],
    [0.3, 0.8, 0.1, 0.5],
    [0.9, 0.3, 0.6, 0.2]
])

# Labels: Class 0, 1, or 2
y = np.array([0, 1, 2, 1])
y_encoded = one_hot(y, 3)  # Convert to one-hot

# Network Parameters
input_neurons = X.shape[1]  
   # 4
print(input_neurons)
hidden_neurons = 100
output_neurons = 3
epochs = 1000
learning_rate = 0.01

# Weight Initialization
np.random.seed(42)
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.1
print(W1)
b1 = np.zeros((1, hidden_neurons))
print(b1)
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.1
print(W2)
b2 = np.zeros((1, output_neurons))
print(b2)

# Training Loop
for epoch in range(epochs):
    # FORWARD PASS
    hidden_input = np.dot(X, W1) + b1
    hidden_output = relu(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = softmax(final_input)

    # LOSS
    loss = cross_entropy(output, y_encoded)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # BACKWARD PASS
    error_output = output - y_encoded  # dL/dZ2
    dW2 = np.dot(hidden_output.T, error_output)
    db2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, W2.T) * relu_derivative(hidden_input)
    dW1 = np.dot(X.T, error_hidden)
    db1 = np.sum(error_hidden, axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# PREDICT FUNCTION
def predict(X):
    hidden_output = relu(np.dot(X, W1) + b1)
    final_output = softmax(np.dot(hidden_output, W2) + b2)
    return np.argmax(final_output, axis=1)

# Test Predictions
print("Predictions:", predict(X))
