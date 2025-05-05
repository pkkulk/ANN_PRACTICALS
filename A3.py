import numpy as np  # Import numpy for arrays and math

# Step 1: Convert a number to its 8-bit ASCII binary representation
def to_ascii_binary(n):
    ascii_code = ord(str(n))  # Convert number to string, then to ASCII
    binary_string = format(ascii_code, '08b')  # Convert ASCII to 8-bit binary
    binary_list = []  # List to hold binary digits

    for bit in binary_string:
        binary_list.append(int(bit))  # Convert each character to integer

    return binary_list  # Example: 0 -> [00110000]

# Step 2: Prepare input data (X) and labels (y)
X = []  # Input features
y = []  # Labels (even = 0, odd = 1)

for i in range(10):
    binary_input = to_ascii_binary(i)  # Convert digit to binary
    X.append(binary_input)             # Add to inputs
    if i % 2 == 0:
        y.append(0)  # Even
    else:
        y.append(1)  # Odd

X = np.array(X)
y = np.array(y)

# Step 3: Initialize the perceptron
weights = np.zeros(8)      # 8 weights for 8-bit input
bias = 0                   # Bias term
learning_rate = 0.1        # Learning rate

# Step 4: Train the perceptron
for epoch in range(10):  # Repeat 10 times
    for i in range(len(X)):
        input_bits = X[i]  # One input sample
        target = y[i]      # Correct label

        # Weighted sum: z = w·x + b
        weighted_sum = np.dot(input_bits, weights) + bias

        # Step activation function
        prediction = 1 if weighted_sum >= 0 else 0

        # Calculate error
        error = target - prediction

        # Update weights and bias
        for j in range(len(weights)):
            weights[j] += learning_rate * error * input_bits[j]
        bias += learning_rate * error

# ✅ Step 5: Define a predict() function
def predict(input_bits, weights, bias):
    weighted_sum = 0
    for j in range(len(weights)):
        weighted_sum += input_bits[j] * weights[j]
    weighted_sum += bias
    return 1 if weighted_sum >= 0 else 0

# ✅ Step 6: Test the trained model
print("Testing on digits 0 to 9:")
for i in range(10):
    input_bits = to_ascii_binary(i)
    prediction = predict(input_bits, weights, bias)
    label = "Odd" if prediction == 1 else "Even"
    print(f"Digit {i} is predicted as: {label}")
