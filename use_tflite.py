import numpy as np
import tensorflow as tf
from utils.data_loader import load_data, show_image


def load_tflite_model(tflite_model_path):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def run_tflite_inference(interpreter, input_details, output_details, input_data):
    # Ensure the input data matches the expected shape and type
    input_shape = input_details[0]["shape"]
    input_data = np.array(input_data, dtype=input_details[0]["dtype"])
    input_data = (
        np.expand_dims(input_data, axis=0)
        if len(input_data.shape) == len(input_shape) - 1
        else input_data
    )

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])

    return output_data


# Example Usage:
tflite_model_path = "trained_models/M3.tflite"
interpreter, input_details, output_details = load_tflite_model(tflite_model_path)

test_path = "data/mnist_test.csv"
test_images, test_labels = load_data(test_path, shuffle=False)

image_number = 1

image = test_images[image_number]
image = np.expand_dims(image, axis=0)

print(image)

output = run_tflite_inference(
    interpreter, input_details, output_details, image.astype(np.float32)
)

predicted_label = np.argmax(output[0])
confidence = np.max(output[0])

print("Real Label:", test_labels[image_number])
print("Predicted Label:", predicted_label)
print("Confidence:", confidence)
