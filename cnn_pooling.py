import numpy as np
import matplotlib.pyplot as plt

# Original matrix
matrix = np.array([
    [12, 20, 30, 0],
    [8, 12, 2, 0],
    [34, 70, 37, 4],
    [112, 100, 25, 12]
])

# Perform max pooling
max_pooled = np.array([
    [np.max(matrix[:2, :2]), np.max(matrix[:2, 2:])],
    [np.max(matrix[2:, :2]), np.max(matrix[2:, 2:])]
])

# Perform average pooling
avg_pooled = np.array([
    [np.mean(matrix[:2, :2]), np.mean(matrix[:2, 2:])],
    [np.mean(matrix[2:, :2]), np.mean(matrix[2:, 2:])]
])

# Plotting the matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
cmap = "cool"

# Original Matrix
axes[0].imshow(matrix, cmap=cmap, interpolation='nearest')
axes[0].set_title("Original Matrix")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        axes[0].text(j, i, matrix[i, j], ha='center', va='center', color='black')

# Max Pooled Matrix
axes[1].imshow(max_pooled, cmap=cmap, interpolation='nearest')
axes[1].set_title("Max Pooling")
for i in range(max_pooled.shape[0]):
    for j in range(max_pooled.shape[1]):
        axes[1].text(j, i, int(max_pooled[i, j]), ha='center', va='center', color='black')

# Average Pooled Matrix
axes[2].imshow(avg_pooled, cmap=cmap, interpolation='nearest')
axes[2].set_title("Average Pooling")
for i in range(avg_pooled.shape[0]):
    for j in range(avg_pooled.shape[1]):
        axes[2].text(j, i, int(avg_pooled[i, j]), ha='center', va='center', color='black')

# Adjust layout
plt.tight_layout()
plt.show()
