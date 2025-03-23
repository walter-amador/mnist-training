# generate a rely graph with range from -10 to 10
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


x = np.linspace(-1, 1, 100)
y = relu(x)

plt.plot(x, y)
plt.title("ReLU Function")
plt.grid()
plt.show()
