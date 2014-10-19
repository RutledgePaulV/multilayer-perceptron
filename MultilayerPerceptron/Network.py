from Utilities import *
from Layers import *
from sys import float_info

class Network(object):
    # constructor with default params
    def __init__(self,
                 num_input,
                 num_hidden,
                 num_out,
                 learning_rate=0.5,
                 momentum=0.0,
                 random=Standard(),
                 activation=Sigmoid()):

        # copying arguments to fields
        self.momentum = momentum
        self.random = random
        self.learning_rate = learning_rate
        self.activation = activation

        # initializing error
        self.error = float_info.max

        # setting up bias neuron
        self.bias = InputNeuron()
        self.bias.input = 1.0
        self.bias.output = 1.0

        # allocating lists
        self.layers = []
        self.hidden_layers = []

        # creating input layer
        self.input_layer = InputLayer(num_input, self.activation)
        self.layers.append(self.input_layer)

        # creating hidden layers
        for x in num_hidden:
            hidden = HiddenLayer(x, self.activation)
            self.hidden_layers.append(hidden)
            self.layers.append(hidden)

        # creating output layer
        self.output_layer = OutputLayer(num_out, self.activation)
        self.layers.append(self.output_layer)

        # connecting layers and bias with weights
        for index in xrange(0, len(self.layers) - 1):
            self.layers[index].connect_child(self.layers[index + 1], self.random)
            self.layers[index + 1].connect_bias(self.bias, self.random)

    def train(self,data,epochs=1,min_error=0.0):

        while epochs > 0 and self.error>min_error:
            self.error = 0.0
            for (input_list,output_list) in data:
                self.input_layer.set_input(input_list)
                self.forward_propagate()
                self.output_layer.assign_errors(output_list)
                self.backward_propagate()
                self.error += self.output_layer.squared_error
            epochs -= 1

    def test(self,input):
        self.input_layer.set_input(input)
        self.forward_propagate()
        return self.output_layer.extract_outputs()

    def forward_propagate(self):

        # looping through every layer except first
        for layer in self.layers[1:]:

            # looping through each neuron in layer
            for neuron in layer.neurons:

                # setting neuron's input to zero
                neuron.input = 0.0

                # for each weight from neuron's parents to itself
                for w in neuron.weights_to_parent:
                    # add parents output times the value of the weight
                    neuron.input += w.parent.output * w.value

                # add contributing value from bias
                neuron.input += neuron.weight_to_bias.parent.output * neuron.weight_to_bias.value

                # transform input through activation function and assign to output
                neuron.output = layer.activation.evaluate(neuron.input)


    # future things to try are tracking more than just the previous value
    # and adjusting by some weighted sum of the previous values
    # maybe determine coefficients by fibonacci series
    def backward_propagate(self):

        # looping through layers backwards and excluding input layer
        for index, layer in enumerate(self.layers[::-1][:-1]):

            # for each neuron in layer
            for neuron in layer.neurons:

                # if not on output layer
                if index != 0:

                    # reset neurons error to zero
                    neuron.error = 0.0

                    # propagate error backwards from the neuron's children
                    for w in neuron.weights_to_child:
                        neuron.error += w.value * w.child.error * w.child.primed

                # getting derivative value of the neurons input
                neuron.primed = layer.activation.evaluate_derivative(neuron.input)

                # holding in a temporary variable to minimize calculations
                base_adjust = self.learning_rate * neuron.error * neuron.primed

                # for each weight from neuron to its parents
                for w in neuron.weights_to_parent:

                    # calculate adjustment factor
                    adjust = base_adjust * w.parent.output

                    # adjust weights value by current adjustment factor and fraction of previous adjustment
                    w.value += adjust + self.momentum * w.previous

                    # setting value of previous adjustment
                    w.previous = adjust

                # calculating amount to adjust bias weight by
                bias_adjust = base_adjust * neuron.weight_to_bias.parent.output

                # adjusting bias weight by current adjustment factor and fraction of previous adjustment
                neuron.weight_to_bias.value += bias_adjust + self.momentum * neuron.weight_to_bias.previous

                # setting value of previous bias adjustment
                neuron.weight_to_bias.previous = bias_adjust

