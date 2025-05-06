import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = np.array(p)
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)  # No self-connection
        self.weights = self.weights / self.size  # Optional scaling

    def recall(self, pattern, steps=5):
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
            pattern[pattern == 0] = 1  # Optional: treat 0 as +1
        return pattern

# 🧠 Define binary patterns to store (using bipolar representation: -1 and +1)
patterns = [
    [1, -1, 1, -1],
    [-1, -1, 1, 1],
    [1, 1, -1, -1],
    [-1, 1, -1, 1]
]

# 🚀 Create and train the Hopfield network
hopfield = HopfieldNetwork(size=4)
hopfield.train(patterns)

# 🔁 Test recall from a noisy input (or same input)
test_pattern = [1, -1, 1, -1]  # Try also noisy inputs like [1, 1, 1, -1]
output = hopfield.recall(test_pattern)

print("Input Pattern :", test_pattern)
print("Recalled Pattern:", output.tolist())
