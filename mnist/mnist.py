import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.keras import layers, models
# Load data stored in numpy format
Data = np.load("Handwritten_data2.npz")
print(Data.files)

y = Data['labels']
m = y.size
X = Data['data']
print(X.shape)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize the pixel values of the images to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Flatten the 28x28 images to a 1D vector
    layers.Dense(128, activation='relu'),      # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax')     # Output layer with 10 neurons (for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Predict on the test data
predictions = model.predict(x_test)

# Visualize the predictions
def plot_images(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"Predicted: {predicted_label} ({100*np.max(predictions_array):2.0f}%) (True: {true_label})", color=color)

# Plot the first 5 test images along with their predictions
for i in range(5):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_images(predictions[i], y_test[i], x_test[i])
    plt.show()
model.save('mnist_model.h5')
