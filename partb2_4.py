import numpy as np
import matplotlib.pyplot as plt

# Define the data
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321])

# Define the neural network architecture
input_size = 1
hidden_size = 10
output_size = 1

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(hidden_size, input_size)
b1 = np.random.randn(hidden_size, 1)
W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size, 1)

# Set hyperparameters
learning_rate = 0.01
epochs = [10, 100, 200, 400, 1000]
errors = []

# Training loop
for epoch in range(max(epochs) + 1):
    # Forward propagation
    Z1 = np.dot(W1, X.reshape(1, -1)) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2

    # Calculate the error
    error = 0.5 * np.mean((A2 - Y) ** 2)
    errors.append(error)

    # Backpropagation
    dZ2 = A2 - Y
    dW2 = (1 / len(X)) * np.dot(dZ2, A1.T)
    db2 = (1 / len(X)) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = (1 / len(X)) * np.dot(dZ1, X.reshape(1, -1).T)
    db1 = (1 / len(X)) * np.sum(dZ1, axis=1, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Plot results for specified epochs
    if epoch in epochs:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(X, Y, label="Actual Function")
        plt.plot(X, A2.flatten(), label="NN Output")
        plt.xlabel("x")
        plt.ylabel("f(x) / NN Output")
        plt.legend()
        plt.title(f'Epoch {epoch}')

# Plot training error vs. epoch number
plt.figure()
plt.plot(range(len(errors)), errors)
plt.xlabel("Epoch")
plt.ylabel("Training Error")
plt.show()
