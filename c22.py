import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Step 2: Preprocess the data
train_images = train_images / 255.0  # Normalize to [0, 1]
test_images = test_images / 255.0

# Add channel dimension (needed for CNN)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Step 3: Build the CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digits (0–9)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_images, train_labels, epochs=3, validation_split=0.1)

# Step 6: Predict and show a test image
index = 0  # Change this to test different images
image = test_images[index]
true_label = test_labels[index]

# Predict
prediction = model.predict(image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

# Show image and prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {true_label}")
plt.axis('off')
plt.show()
