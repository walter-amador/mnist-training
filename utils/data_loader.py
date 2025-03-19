import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename, shuffle=True):
    data = pd.read_csv(filename)
    data = np.array(data)
    m, n = data.shape
    if shuffle:
        np.random.shuffle(data)
    data = data.T
    labels = data[0]
    images = data[1:n].T.reshape(-1, 28, 28, 1)
    images = images / 255.
    return images, labels

def show_image(image, label):
    image = image.reshape(28, 28)
    plt.title(f"Label: {label}")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
