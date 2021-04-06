# create the csv file
# import csv
# with open('XOR_tabel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["x1", "x2", "y"])
#     writer.writerow([0,0,0])
#     writer.writerow([0,1,1])
#     writer.writerow([1,0,1])
#     writer.writerow([1,1,0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def main():
    XOR_tabel = pd.read_csv('XOR_tabel.csv')
    # print(XOR_tabel)

    XOR_tabel = XOR_tabel.values
    # print(XOR_tabel)
    X = XOR_tabel[:, :2]
    targets = XOR_tabel[:, -1].reshape(-1, 1)

    # print("X: ",X)
    # print("targets: ",targets)

    # Define dimensions on input, hidden and output layers
    input_dim, hidden_dim, output_dim = 2, 16, 1

    num_iteration = 10000
    # Define learning rate
    learning_rate = 0.01

    # Define a hidden layer
    W1 = np.random.normal(size=[input_dim, hidden_dim])
    print(W1.shape)
    # Define an output layer
    W2 = np.random.normal(size=[hidden_dim, output_dim])
    print(W2.shape)

    # training
    loss_log = []
    for i in range(10000):
        # Forward pass: compute predicted y
        z = sigmoid(np.matmul(X, W1))
        y = sigmoid(np.matmul(z, W2))
        # print(y, targets_one_hot)

        # Compute and print loss
        Loss = np.sum((y - targets)**2)

        # Backprop to compute gradients of w1 and w2 w.r.t L2-norm Loss
        temp = 2 * (y - targets) * y * (1-y)
        aJaW2 = np.matmul(np.transpose(z), temp)
        a = np.transpose(W2) * z * (1 - z)
        aJaW1 = np.matmul(np.transpose(X), a * temp)
        # Update weights
        W2 = W2 - learning_rate * aJaW2
        W1 = W1 - learning_rate * aJaW1
        # Save loss to an array
        loss_log.append(Loss)
        if i % 1000 == 0:
            print(i, Loss)

    z = sigmoid(np.matmul(X, W1))
    final_output = sigmoid(np.matmul(z, W2))
    print(final_output)
    # print(loss_log)
    plt.figure()
    plt.plot(loss_log)
    plt.show()

if __name__ == "__main__":
    main()