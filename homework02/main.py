import numpy as np
from numpy import random
from layer import Layer
from mlp import MLP
from matplotlib import pyplot as plt

# Task 01 - Building the data set
x = np.random.uniform(0, 5, 10)
t = np.empty(10)

for i in range(len(x)):
    tmp = (x[i] ** 2)
    t[i] = tmp


def shuffle_function(a, b):
    assert len(a) == len(b)
    shuffle = np.random.permutation(len(a))
    return a[shuffle], b[shuffle]


# task 4.1: create layers and mlp
# creating a hidden layer with 10 units and 1 input and an output layer with 1 unit and 10 inputs
layers = [Layer(10, 1), Layer(1, 10)]
# creating a mlp with those layers
mlp = MLP(layers)


def learning_process(mlp, inputs, epochs, targets, learning_rate):
    """
    @param mlp: mlp from mlp class
    @param inputs: input values
    @param epochs: amount of epochs
    @param targets: targets from the input values
    @param learning_rate: a set learning rate
    @return: a nested list with all losses from all inputs in all epochs
    """

    for_mean = []
    # for loop iterating over the epochs
    for i in range(epochs):

        losses = []
        # shuffling the inputs to get a better learning process
        shuffle_function(inputs, targets)

        # for loop iterating over all inputs
        for j in range(len(inputs)):
            # doing this due to dimension problems,that should not effect the result
            dim = [inputs[j]]

            # executing the forward step in mlp class
            mlp.forward_step2(dim)

            # calculating the loss
            loss = 0.5 * (mlp.output - targets[j]) ** 2

            # inserting the losses into an array for the visualization
            losses = np.append(losses, loss)

            # executing the backpropagation in mlp class
            mlp.backpropagation(targets[j], learning_rate)

        losses = np.mean(losses)
        for_mean = np.append(for_mean, losses)

    return for_mean


# executing the learning process with the given values
loss_values = learning_process(mlp, x, 1000, t, 0.04)



# Task 2.5 - Visualization
epoch = np.arange(1, 1001, 1)

plt.plot(epoch, loss_values)
plt.title("Loss Development over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Means")
plt.show()
