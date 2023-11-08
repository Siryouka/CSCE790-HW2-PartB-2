import numpy as np
import matplotlib.pyplot as plt


def dsig(N, A):
    return A * (1 - A)


X = np.array(
    [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array(
    [-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396,
     0.345, 0.182, -0.031, -0.219, -0.321])
input_layer = 1
output_layer = 1
h = 30

U = np.random.randn(input_layer, h)
U0 = np.random.rand(h, 1)
V = np.random.randn(h, output_layer)
V0 = np.random.rand(output_layer, 1)

lr = 0.06
E = []
y = []
epochs = [10, 100, 200, 400, 1000]

# # Store the results for different epochs
# results = {}
y = np.zeros((1, 21))

for epoch in range(max(epochs) + 1):
    loss_all = 0
    for i in range(len(X)):
        del_1 = np.matmul(U.T, np.reshape(X[i], (1, 1))) + U0

        z1 = 1 / (1 + np.exp(-del_1))  # ?
        del_2 = np.matmul(V.T, z1) + V0
        y[:, i] = del_2
        e = Y[i] - y[:, i]

        loss_all += e ** 2

        V = V + lr * np.matmul(z1, np.reshape(e.T, (1, 1)))
        V0 = V0 + lr * e.T
        t = (dsig(del_1, z1) * np.matmul(V, np.reshape(e, (1, 1)))).T
        U = U + lr * np.matmul(np.reshape(X[i].T, (1, 1)), t)
        U0 = U0 + lr * np.ones((h, 1)) * (dsig(del_1, z1) * np.matmul(V, np.reshape(e, (1, 1))))
    E.append(loss_all / 21)
    if (epoch == 10):
        plt.figure(1)
        plt.plot(range(1, 22), y[0, :], 'r', range(1, 22), Y)
        plt.title(f'Epoch {epoch}')
        plt.show()

    if (epoch == 100):
        plt.figure(2)
        plt.plot(range(1, 22), y[0, :], 'r', range(1, 22), Y)
        plt.title(f'Epoch {epoch}')
        plt.show()

    if (epoch == 200):
        plt.figure(3)
        plt.plot(range(1, 22), y[0, :], 'r', range(1, 22), Y)
        plt.title(f'Epoch {epoch}')
        plt.show()

    if (epoch == 400):
        plt.figure(4)
        plt.plot(range(1, 22), y[0, :], 'r', range(1, 22), Y)
        plt.title(f'Epoch {epoch}')
        plt.show()

    if (epoch == 1000):
        plt.figure(5)
        plt.plot(range(1, 22), y[0, :], 'r', range(1, 22), Y)
        plt.title(f'Epoch {epoch}')
        plt.show()


plt.figure(6)
plt.plot(range(len(E)), E)
plt.xlabel("Epoch")
plt.ylabel("Training Error")
plt.show()
# hold()
# plt.hold(True)
# plt.plot(Y, 'b')
# plt.show()
# plt.hold(False)


# Plot training error vs. epoch number