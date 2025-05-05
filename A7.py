import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (XOR inputs)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output (XOR outputs)
expected_output = np.array([[0], [1], [1], [0]]) 

# Set seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_layer_neurons = inputs.shape[1]  # 2
hidden_layer_neurons = 4               # You can change this
output_neurons = 1

# Random weights and biases
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
print("hide weight->",hidden_weights)
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
print("hide bias->",hidden_bias)
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
print("out weights->",output_weights)
output_bias = np.random.uniform(size=(1, output_neurons))
print("out bias->",output_bias)
# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Optionally print loss
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Final output after training
print("\nFinal predictions after training:")
print(predicted_output.round(2))
