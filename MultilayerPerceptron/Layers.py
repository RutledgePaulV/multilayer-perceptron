from Activation import *
from Neurons import *


class BaseLayer(object):
    def __init__(self, size=0, activation=Sigmoid()):
        self.size = size
        self.neurons = []
        self.activation = activation

    def connect_parent(self, layer, random):
        for child in self.neurons:
            for parent in layer.neurons:
                w = Weight(random.generate(), parent, child)
                child.weights_to_parent.append(w)
                parent.weights_to_child.append(w)

    def connect_child(self, layer, random):
        for parent in self.neurons:
            for child in layer.neurons:
                w = Weight(random.generate(), parent, child)
                parent.weights_to_child.append(w)
                child.weights_to_parent.append(w)

    def connect_bias(self, bias, random):
        for child in self.neurons:
            w = Weight(random.generate(), bias, child)
            child.weight_to_bias = w
            bias.weights_to_child.append(w)


class InputLayer(BaseLayer):
    def __init__(self, size=0, activation=Sigmoid()):
        super(InputLayer, self).__init__(size, activation)
        for x in xrange(0, size):
            self.neurons.append(InputNeuron())

    def set_input(self, input):
        for index, neuron in enumerate(self.neurons):
            neuron.input = input[index]
            neuron.output = input[index]


class HiddenLayer(BaseLayer):
    def __init__(self, size=0, activation=Sigmoid()):
        super(HiddenLayer, self).__init__(size, activation)
        for x in xrange(0, size):
            self.neurons.append(HiddenNeuron())


class OutputLayer(BaseLayer):
    def __init__(self, size=0, activation=Sigmoid()):
        super(OutputLayer, self).__init__(size, activation)
        for _ in xrange(0, size):
            self.neurons.append(OutputNeuron())

    def assign_errors(self, expected):
        for index, neuron in enumerate(self.neurons):
            neuron.error = expected[index] - neuron.output

    def extract_outputs(self):
        return [x.output for x in self.neurons]

    @property
    def squared_error(self):
        return sum([(n.error ** 2) for n in self.neurons]) / 2
