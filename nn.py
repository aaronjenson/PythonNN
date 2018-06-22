import numpy as np
import random


def sigmoid(x):
    """
    activation function
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    """
    derivative of activation function
    """
    return y * (1.0 - y)


class NeuralNetwork(object):
    """
    universal function approximater
    """

    def __init__(self, *layers):
        """
        :param layers: number of nodes in each layer
        """

        # check that there is at least an input and output layer
        if len(layers) < 2:
            raise ValueError("Not enough layers!")

        # convert layers tuple to list, add bias
        self.layers = list(layers)
        self.layers[0] += 1
        # save some values for easier use later
        self.num_layers = len(self.layers)
        self.inputs = self.layers[0]
        self.outputs = self.layers[self.num_layers - 1]

        # create activation lists
        self.activations = []
        for i in self.layers:
            self.activations.append([1.0] * i)

        # create weights matrices w/ random values
        self.weights = [None] * self.num_layers
        for i in range(1, self.num_layers):
            self.weights[i] = np.random.randn(self.layers[i - 1], self.layers[i])

        # create change matrices of zeros
        self.changes = [None] * self.num_layers
        for i in range(1, self.num_layers):
            self.changes[i] = np.zeros((self.layers[i - 1], self.layers[i]))

    def feed_forward(self, inputs):
        """
        calculate a prediction based off the inputs
        :param inputs: list of inputs
        :return: list of outputs
        """

        # check that the number of inputs is correct
        if len(inputs) != self.inputs - 1:
            raise ValueError('Wrong number of inputs')

        # set input activations to inputs
        for i in range(self.inputs - 1):
            self.activations[0][i] = inputs[i]

        # calculate activations of each layer
        for i in range(1, self.num_layers):
            for j in range(self.layers[i]):
                sum = 0.0
                for k in range(self.layers[i - 1]):
                    sum += self.activations[i - 1][k] * self.weights[i][k][j]
                self.activations[i][j] = sigmoid(sum)

        # return the output layer's activations
        return self.activations[len(self.activations) - 1][:]

    def back_propagate(self, targets, learning_rate):
        """
        change weights based off the difference between activations and targets
        must be called after feed_forward
        :param targets: list of targets, same size as outputs
        :param learning_rate: learning rate
        :return: current total error
        """

        # check that number of targets is same as number of outputs
        if len(targets) != self.outputs:
            raise ValueError('Wrong number of targets')

        # initialize delta lists for each layer, except inputs
        deltas = [None] * self.num_layers
        for i in range(1, self.num_layers):
            deltas[i] = [0.0] * self.layers[i]

        # initialize output deltas based on targets
        for i in range(self.outputs):
            error = -(targets[i] - self.activations[len(self.activations) - 1][i])
            deltas[self.num_layers - 1][i] = dsigmoid(self.activations[len(self.activations) - 1][i]) * error

        # calculate deltas for each layer
        for i in range(self.num_layers - 2, 0, -1):
            for j in range(self.layers[i]):
                error = 0.0
                for k in range(self.layers[i + 1]):
                    error += deltas[i + 1][k] * self.weights[i + 1][j][k]
                deltas[i][j] = dsigmoid(self.activations[i][j]) * error

        # change the weights based off the deltas
        for i in range(self.num_layers - 2, -1, -1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i + 1]):
                    change = deltas[i + 1][k] * self.activations[i][j]
                    self.weights[i + 1][j][k] -= learning_rate * change + self.changes[i + 1][j][k]
                    self.changes[i + 1][j][k] = change

        # calculate and return total error
        error = 0.0
        for i in range(len(targets)):
            error += 0.5 * (targets[i] - self.activations[len(self.activations) - 1][i]) ** 2
        return error

    def train(self, patterns, iterations=3000, learning_rate=0.0002):
        """
        trains network based off the patterns given
        :param patterns: list of lists containing lists for inputs and outputs of training data
        :param iterations: number of times to cycle through the patterns, default=3000
        :param learning_rate: learning rate, default=0.0002
        """

        # loop through specified number of times,
        # multiplied by number of patterns, since each loop trains on only one pattern
        for i in range(iterations * len(patterns)):
            p = random.choice(patterns)
            inputs = p[0]
            targets = p[1]
            self.feed_forward(inputs)
            error = self.back_propagate(targets, learning_rate)
            if i % (500 * len(patterns)) == 0:
                print('error %-.5f' % error)
