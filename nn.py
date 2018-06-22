import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class NeuralNetwork(object):
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
        self.inputs = self.layers[0]
        self.outputs = self.layers[len(self.layers) - 1]
        # self.input = layers[0] + 1
        # self.hidden = layers[1]
        # self.output = layers[2]

        # create activation lists
        self.activations = []
        for i in self.layers:
            self.activations.append([1.0] * i)
        # self.ai = [1.0] * self.input
        # self.ah = [1.0] * self.hidden
        # self.ao = [1.0] * self.output

        # create weights matrices w/ random values
        self.weights = [None] * len(self.layers)
        for i in range(1, len(self.layers)):
            self.weights[i] = np.random.randn(self.layers[i - 1], self.layers[i])
        # self.wi = np.random.randn(self.input, self.hidden)
        # self.wo = np.random.randn(self.hidden, self.output)

        # create change matrices of zeros
        self.changes = [None] * len(self.layers)
        for i in range(1, len(self.layers)):
            self.changes[i] = np.zeros((self.layers[i - 1], self.layers[i]))
        # self.ci = np.zeros((self.input, self.hidden))
        # self.co = np.zeros((self.hidden, self.output))

    def feed_forward(self, inputs):
        if len(inputs) != self.inputs - 1:
            raise ValueError('Wrong number of inputs')

        # for each input
        for i in range(self.inputs - 1):
            # set input_activation to input
            self.activations[0][i] = inputs[i]

        # for each layer except input
        for i in range(1, len(self.layers)):
            # for each node in that layer
            for j in range(self.layers[i]):
                sum = 0.0
                # for each node in the previous layer
                for k in range(self.layers[i - 1]):
                    # add the previous layer's node's
                    sum += self.activations[i - 1][k] * self.weights[i][k][j]
                self.activations[i][j] = sigmoid(sum)

        return self.activations[len(self.activations) - 1][:]

        # for i in range(self.input - 1):
        #     self.ai[i] = inputs[i]
        #
        # for j in range(self.hidden):
        #     sum = 0.0
        #     for i in range(self.input):
        #         sum += self.ai[i] * self.wi[i][j]
        #     self.ah[j] = sigmoid(sum)
        #
        # for k in range(self.output):
        #     sum = 0.0
        #     for j in range(self.hidden):
        #         sum += self.ah[j] * self.wo[j][k]
        #     self.ao[k] = sigmoid(sum)
        #
        # return self.ao[:]

    def back_propagate(self, targets, N):
        if len(targets) != self.layers[len(self.layers) - 1]:
            raise ValueError('Wrong number of targets')

        deltas = [None] * len(self.layers)
        # for each layer, except input layer
        for i in range(1, len(self.layers)):
            # initialize deltas array with zeroes
            deltas[i] = [0.0] * self.layers[i]

        # for each output
        for i in range(self.outputs):
            # calculate error (target - output activation)
            error = -(targets[i] - self.activations[len(self.activations) - 1][i])
            deltas[len(self.layers) - 1][i] = dsigmoid(self.activations[len(self.activations) - 1][i]) * error

        # for each layer, except output and input, in reverse order
        for i in range(len(self.layers) - 2, 0, -1):
            # for each node in layer
            for j in range(self.layers[i]):
                error = 0.0
                # for each node in next layer
                for k in range(self.layers[i + 1]):
                    error += deltas[i + 1][k] * self.weights[i + 1][j][k]
                deltas[i][j] = dsigmoid(self.activations[i][j]) * error

        # for each layer, except output, in reverse order
        for i in range(len(self.layers) - 2, -1, -1):
            # for each node in layer
            for j in range(self.layers[i]):
                # for each node in next layer
                for k in range(self.layers[i + 1]):
                    change = deltas[i + 1][k] * self.activations[i][j]
                    self.weights[i + 1][j][k] -= N * change + self.changes[i + 1][j][k]
                    self.changes[i + 1][j][k] = change

        error = 0.0

        for i in range(len(targets)):
            error += 0.5 * (targets[i] - self.activations[len(self.activations) - 1][i]) ** 2

        return error

        # output_deltas = [0.0] * self.output
        #
        # for k in range(self.output):
        #     error = -(targets[k] - self.ao[k])
        #     output_deltas[k] = dsigmoid(self.ao[k]) * error
        #
        # hidden_deltas = [0.0] * self.hidden
        # for j in range(self.hidden):
        #     error = 0.0
        #     for k in range(self.output):
        #         error += output_deltas[k] * self.wo[j][k]
        #     hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        #
        # for j in range(self.hidden):
        #     for k in range(self.output):
        #         change = output_deltas[k] * self.ah[j]
        #         self.wo[j][k] -= N * change + self.co[j][k]
        #         self.co[j][k] = change
        #
        # for i in range(self.input):
        #     for j in range(self.hidden):
        #         change = hidden_deltas[j] * self.ai[i]
        #         self.wi[i][j] -= N * change + self.ci[i][j]
        #         self.ci[i][j] = change
        #
        # error = 0.0
        #
        # for k in range(len(targets)):
        #     error += 0.5 * (targets[k] - self.ao[k]) ** 2
        #
        # return error

    def train(self, patterns, iterations=3000, N=0.0002):
        for i in range(iterations * len(patterns)):
            error = 0.0
            p = random.choice(patterns)
            inputs = p[0]
            targets = p[1]
            self.feed_forward(inputs)
            error = self.back_propagate(targets, N)
            if i % (500 * len(patterns)) == 0:
                print('error %-.5f' % error)

    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feed_forward(p))
        return predictions
