import numpy as np
from numpy import random
from layer import Layer


# Task 2.3 createing a mlp class
class MLP:

    def __init__(self, layers):
        """
        @param layers: a list of layers the mlp is build up of
        """
        self.layers = layers
        self.output = np.nan

    # task 2.3.1: forward_step in mlp
    def forward_step2(self, input):
        """
        @param input: input of the current layer
        @return: nothing, all things that need to be saved/changed are executed or saved in the object itself
        """
        act = input
        # for loop iterating over all layers of the mlp
        for i in range(len(self.layers)):
            # if the output layer is reached, the output is saved in the mlp class as well as
            # the layer class and later used for further calculations
            if i is (len(self.layers) - 1):
                self.output = self.layers[i].forward_step(act)  #####
                self.layers[i].layer_activation = self.output
                return

            # if the output layer is not reached yet, we just execute the forward step in layer class
            else:
                act = self.layers[i].forward_step(act)
                self.layers[i].layer_activation = act
                # the result is saved as the input of the next layer for simplification reasons
                self.layers[i + 1].layer_input = act

    # task 2.3.2: backpropagation
    def backpropagation(self, target, learning_rate):
        """
        @param target: target value for input
        @param learning_rate: learning rate that was chosen at the beginning
        @return: nothing, all things that need to be saved/changed are executed or saved in the object itself
        """
        # the first delta cann be calculated with the derivative of the loss function
        delta = [(self.output - target)]

        # the for loop iterates over all layers in a revered way to do backpropagation
        for values in reversed((self.layers)):
            # the backward step from layer class is executed
            values.backward_step(delta, learning_rate)

            # the delta is calculated and saved for the next iteration
            values.delta = delta[0] * values.weight_matrix
            delta = values.delta

            return
