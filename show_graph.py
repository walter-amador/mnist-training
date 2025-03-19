import numpy as np
import matplotlib.pyplot as plt

model_name = "M3"

# Load history
history = np.load(f"./diagrams_info/{model_name}_history.npy", allow_pickle=True).item()
print(history.keys())

accuracy = history["accuracy"]
accuracy = np.array(accuracy)
print(accuracy.shape)
accuracy = accuracy * 100
val_accuracy = history["val_accuracy"]
val_accuracy = np.array(val_accuracy)
val_accuracy = val_accuracy * 100
loss = history["loss"]
val_loss = history["val_loss"]

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(accuracy, label="Training set")
plt.plot(val_accuracy, label="Validation set")
plt.title(f"{model_name} accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss, label="Training set")
plt.plot(val_loss, label="Validation set")
plt.title(f"{model_name} loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
