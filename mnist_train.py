# Paper: An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition
# Training data set: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils.data_loader import load_data, show_image
from utils.model import create_cnn

# Define paths
train_path = "data/mnist_train.csv"
test_path = "data/mnist_test.csv"

# Load dataset
train_images, train_labels = load_data(train_path)
test_images, test_labels = load_data(test_path)

train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:100]
test_labels = test_labels[:100]

EPOCHS = 3

models = {
    "M3": create_cnn(kernel_size=3, num_layers=10, channels_per_layer=16),
    "M5": create_cnn(kernel_size=5, num_layers=5, channels_per_layer=32),
    "M7": create_cnn(kernel_size=7, num_layers=4, channels_per_layer=48),
}

print("Successfully created models")

# Show model architecture
for name, model in models.items():
    print(name)
    print(model.summary())

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(
    120
)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(120)


# Training function
def train_model(model, epochs=5, lr=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset, validation_data=test_dataset, epochs=epochs, verbose=1
    )
    return model, history


# Save trained model
def save_model(model, name):
    model.export(f"./{name}")
    print(f"Model {name} saved.")


# Train models
for name, model in models.items():
    print(f"Training {name}...")
    models[name], history = train_model(model, epochs=EPOCHS)

    # Save history
    np.save(f"{name}_history.npy", history.history)

    save_model(models[name], name)

print("Training complete.")
