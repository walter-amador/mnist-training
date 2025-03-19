from tensorflow import keras
from keras import layers

def create_cnn(kernel_size, num_layers, channels_per_layer):
    model = keras.Sequential()

    # Convolutional layers
    for i in range(num_layers):
        filters = channels_per_layer * (i + 1)
        model.add(layers.Conv2D(filters, kernel_size, padding="valid", activation=None))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(10, activation="softmax"))

    return model
