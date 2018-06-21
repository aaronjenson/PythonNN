import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        self.input = input + 1
        self.hidden = hidden
        self.output = output

        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feed_forward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs')

        for i in range(self.input - 1):
            self.ai[i] = inputs[i]

        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def back_propagate(self, targets, N):
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets')

        output_deltas = [0.0] * self.output

        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= N * change + self.co[j][k]
                self.co[j][k] = change

        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= N * change + self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0

        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2

        return error

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