
class BaseNeuron(object):

    def __init__(self):
        self.input = 0.0
        self.output = 0.0
        self.error = 0.0
        self.primed = 0.0

class InputNeuron(BaseNeuron):

    def __init__(self):
        super(InputNeuron,self).__init__()
        self.weights_to_child = []

class HiddenNeuron(BaseNeuron):

    def __init__(self):
        super(HiddenNeuron,self).__init__()
        self.weights_to_child = []
        self.weights_to_parent = []
        self.weight_to_bias = None

class OutputNeuron(BaseNeuron):

    def __init__(self):
        super(OutputNeuron,self).__init__()
        self.weights_to_parent = []
        self.weight_to_bias = None

class Weight(object):

    def __init__(self, value, parent, child):
        self.previous = 0.0
        self.value = value
        self.parent = parent
        self.child = child