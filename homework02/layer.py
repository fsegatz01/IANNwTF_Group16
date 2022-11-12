import numpy as np
from numpy import random
import tensorflow as tf


# the Relu function implemented to be used in the code as activation function
def relu(x):
    return np.maximum(0, x)


# the derivative of Relu function implemented to be used in the code as activation function
def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Task 2:
class Layer:

    def __init__(self, n_units, input_units, bias_vector=None, weight_matrix=None,
                 layer_preactivation=None,
                 layer_activation=None):
        """

        @param n_units: amount of units inside the layer
        @param input_units: amount of input units of the layer before
        @param bias_vector: bias vector of the layer
        @param weight_matrix: weight matrix of the layer
        @param layer_preactivation: preactivation of the layer (summed input before activation function)
        @param layer_activation: activation of the layer (summed input after activation function)
        """
        self.n_units = n_units
        self.input_units = input_units
        self.bias_vector = np.zeros(n_units)
        self.weight_matrix = np.random.randint(0, 10, size=(input_units, n_units))
        self.layer_activation = layer_activation
        self.layer_preactivation = layer_preactivation
        # delta is calculated in the mlp class, but needed for some calculation in this class
        self.delta = np.nan

    # getter method for delta
    def get_delta(self):
        return self.delta

    # setter method for delta
    def set_delta(self, delta):
        self.delta = delta

    # creating a property for delta
    delt = property(get_delta, set_delta)

    # getter method for weight matrix
    def get_weight(self):
        return self.weight_matrix

    # setter method for weight matrix
    def set_weight(self, delta):
        self.weight_matrix = weight_matrix

    # creating a property for delta
    wei = property(get_weight, set_weight)

    # task 2.2.2: forward_step
    def forward_step(self, input):
        """
        @param input: the input that is received from the layer before
        @return: layer activation
        """
        # weight  matrix needed a reshape due to dimensionality errors, that should not effect the result
        self.weight_matrix = self.weight_matrix.reshape(self.input_units, self.n_units)

        # calculation of layer preactivation (forward step 1)
        self.layer_preactivation = (input @ self.weight_matrix) + self.bias_vector

        # calculation of layer activation (forward step 2)
        self.layer_activation = relu(self.layer_preactivation)

        return self.layer_activation

    # tak 2.2.3: backward_step
    def backward_step(self, delta, learning_rate):
        """
        @param delta: calculated delta from another method in mlp class
        @param learning_rate: learning rate that was set in the beginning
        @return: nothing, all things that need to be saved/changed are executed or saved in the object itself
        """
        self.delta = delta

        # saving the first component of the first equation
        r = np.transpose(self.layer_input).reshape(1, self.layer_input.shape[0])
        # saving the second component of the first equation
        k = (derivative_relu(self.layer_preactivation) @ delta)
        # with the two components calculating the gradient
        w_gradient = np.matmul(k, r)

        # reshaping due to dimension problems, that should not effect the result
        self.weight_matrix = self.weight_matrix.reshape(self.weight_matrix.shape[0], )

        # calculating the updated weight and saving it for the  current layer
        self.weight_matrix = self.weight_matrix - (w_gradient * learning_rate)

        # calculating the updated bias vector and saving it for the  current layer
        self.bias_vector = self.bias_vector - (np.dot(delta, learning_rate))
        return
