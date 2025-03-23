import tensorflow as tf

def convert_keras_to_tflite(keras_model_path, tflite_model_path, optimize=False):
    # Load the Keras model
    model = tf.saved_model.load(keras_model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations if required
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_model_path}")

# Example usage:
convert_keras_to_tflite('trained_models/M5', 'trained_models/M5.tflite', optimize=False)
