import nn

data = [[[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 1], [0]],
        [[1, 0], [1]]]


def main():
    neural_network = nn.NeuralNetwork(2, 4, 4, 1)
    neural_network.train(data)

    print('[0, 0] => %-.5f' % neural_network.feed_forward([0, 0])[0])
    print('[0, 1] => %-.5f' % neural_network.feed_forward([0, 1])[0])
    print('[1, 1] => %-.5f' % neural_network.feed_forward([1, 1])[0])
    print('[1, 0] => %-.5f' % neural_network.feed_forward([1, 0])[0])


if __name__ == '__main__':
    main()
