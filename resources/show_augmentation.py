# in a graph show 1 image
# next to the original image show a translated version of the image
# next to the translated image show a rotated version of the image
# next to the rotated image show a scaled version of the image
# next to the scaled image show a flipped version of the image
# next to the flipped image show a noisy version of the image
# next to the noisy image show a blurred version of the image
# next to the blurred image show a sharpened version of the image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import load_data

# Define paths
test_path = "data/mnist_test.csv"

# Load dataset (Data is coming shuffled)
test_images, test_labels = load_data(test_path, shuffle=False)
print(test_images.shape, test_labels.shape)

# transform image
def translate(image, shift):
    return np.roll(image, shift, axis=1)

def rotate(image, angle):
    return np.rot90(image, k=angle//90)

def scale(image, scale):
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

def flip(image):
    return np.flip(image, axis=1)

def noise(image, noise_level):
    return np.clip(image + noise_level * np.random.randn(*image.shape), 0, 1)

def blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    return np.clip(np.convolve(image.ravel(), kernel.ravel(), mode="same").reshape(image.shape), 0, 1)

def sharpen(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    blurred = np.convolve(image.ravel(), kernel.ravel(), mode="same").reshape(image.shape)
    return np.clip(2 * image - blurred, 0, 1)

# Show 1 image
idx = np.random.randint(0, test_images.shape[0])
image = test_images[idx].reshape(28, 28)
label = test_labels[idx]

# Show 1 image in a 2x4 grid
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

# Show original image
axes[0].imshow(image, cmap="gray")
axes[0].set_title(str(label))
axes[0].axis("off")

# Show translated image
shift = 3
translated_image = translate(image, shift)
axes[1].imshow(translated_image, cmap="gray")
axes[1].set_title("Translated")
axes[1].axis("off")

# Show rotated image
angle = 90
rotated_image = rotate(image, angle)
axes[2].imshow(rotated_image, cmap="gray")
axes[2].set_title("Rotated")
axes[2].axis("off")

# Show scaled image
scale_factor = 2
scaled_image = scale(image, scale_factor)
axes[3].imshow(scaled_image, cmap="gray")
axes[3].set_title("Scaled")
axes[3].axis("off")

# Show flipped image
flipped_image = flip(image)
axes[4].imshow(flipped_image, cmap="gray")
axes[4].set_title("Flipped")
axes[4].axis("off")

# Show noisy image
noise_level = 0.1
noisy_image = noise(image, noise_level)
axes[5].imshow(noisy_image, cmap="gray")
axes[5].set_title("Noisy")
axes[5].axis("off")

# Show blurred image
kernel_size = 5
blurred_image = blur(image, kernel_size)
axes[6].imshow(blurred_image, cmap="gray")
axes[6].set_title("Blurred")
axes[6].axis("off")

# Show sharpened image
sharpened_image = sharpen(image, kernel_size)
axes[7].imshow(sharpened_image, cmap="gray")
axes[7].set_title("Sharpened")
axes[7].axis("off")

plt.tight_layout()
plt.show()