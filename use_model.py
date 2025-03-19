import numpy as np
import tensorflow as tf
from utils.data_loader import load_data, show_image

# Load the saved model
model = tf.saved_model.load("trained_models/M3")

# Check available signatures
print("Model signatures:", model.signatures.keys())

# Get the inference function
infer = model.signatures["serving_default"]

# Define paths
test_path = "data/mnist_test.csv"

# Load dataset (Data is coming shuffled)
test_images, test_labels = load_data(test_path, shuffle=False)
print(test_images.shape, test_labels.shape)

image_number = 9990

show_image(test_images[image_number], test_labels[image_number])

print("Real Label:", test_labels[image_number])

# Ensure the input image has the correct shape
image = test_images[image_number]
image = np.expand_dims(image, axis=0)

# Perform inference
output = infer(tf.constant(image.astype(np.float32)))

# Get the predicted label
predicted_label = np.argmax(list(output.values())[0][0])
confidence = np.max(list(output.values())[0][0])

print("Predicted Label:", predicted_label)
print("Confidence:", confidence)
