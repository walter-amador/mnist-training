import numpy as np
import tensorflow as tf
from utils.data_loader import load_data, show_image

# Load the saved model
model_m3 = tf.saved_model.load("trained_models/M3")
model_m5 = tf.saved_model.load("trained_models/M5")
model_m7 = tf.saved_model.load("trained_models/M7")

# Check available signatures
# print("Model signatures M3:", model_m3.signatures.keys())
# print("Model signatures M5:", model_m5.signatures.keys())
# print("Model signatures M7:", model_m7.signatures.keys())

# Get the inference function
infer_m3 = model_m3.signatures["serving_default"]
infer_m5 = model_m5.signatures["serving_default"]
infer_m7 = model_m7.signatures["serving_default"]

# Define paths
test_path = "data/mnist_test.csv"

# Load dataset (Data is coming shuffled)
test_images, test_labels = load_data(test_path, shuffle=False)
print(test_images.shape, test_labels.shape)

image_number = 2

show_image(test_images[image_number], test_labels[image_number])

print("Real Label:", test_labels[image_number])

# Ensure the input image has the correct shape
image = test_images[image_number]
image = np.expand_dims(image, axis=0)

# Perform inference
output_m3 = infer_m3(tf.constant(image.astype(np.float32)))
output_m5 = infer_m5(tf.constant(image.astype(np.float32)))
output_m7 = infer_m7(tf.constant(image.astype(np.float32)))

# Get the predicted label
predicted_label_m3 = np.argmax(list(output_m3.values())[0][0])
predicted_label_m5 = np.argmax(list(output_m5.values())[0][0])
predicted_label_m7 = np.argmax(list(output_m7.values())[0][0])
confidence_m3 = np.max(list(output_m3.values())[0][0])
confidence_m5 = np.max(list(output_m5.values())[0][0])
confidence_m7 = np.max(list(output_m7.values())[0][0])

print("Predicted Label M3:", predicted_label_m3)
print("Confidence M3:", confidence_m3)
print("Predicted Label M5:", predicted_label_m5)
print("Confidence M5:", confidence_m5)
print("Predicted Label M7:", predicted_label_m7)
print("Confidence M7:", confidence_m7)
