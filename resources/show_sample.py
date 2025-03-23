# in a graph show 6 randomly chosen numbers from the test set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import load_data

# Define paths
test_path = "data/mnist_test.csv"

# Load dataset (Data is coming shuffled)
test_images, test_labels = load_data(test_path, shuffle=False)
print(test_images.shape, test_labels.shape)

# Show 6 random images in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(8, 12))
axes = axes.flatten()
for ax in axes:
    idx = np.random.randint(0, test_images.shape[0])
    image = test_images[idx].reshape(28, 28)
    label = test_labels[idx]
    ax.imshow(image, cmap="gray")
    ax.set_title(str(label))
    ax.axis("off")
plt.tight_layout()
plt.show()
