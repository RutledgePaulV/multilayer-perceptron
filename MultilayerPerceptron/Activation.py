import abc
from Utilities import *
from math import *


class BaseActivation(object):
    def __init__(self):
        self.alpha = 1
        self.input_range = None
        self.output_range = None

    @abc.abstractmethod
    def evaluate(self, val):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_derivative(self, val):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_abstract_derivative(self, val):
        raise NotImplementedError


class Sigmoid(BaseActivation):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input_range = Range(float('-inf'), float('inf'))
        self.output_range = Range(0, 1)

    def evaluate(self, val):
        return 1.0 / (1 + exp(-self.alpha * val))

    def evaluate_derivative(self, val):
        temp = exp(self.alpha * val)
        return (self.alpha * temp) / ((temp + 1) * (temp + 1))

    def evaluate_abstract_derivative(self, val):
        return self.alpha * val * (1 - val)


class BipolarSigmoid(BaseActivation):
    def __init__(self):
        super(BipolarSigmoid, self).__init__()
        self.input_range = Range(float('-inf'), float('inf'))
        self.output_range = Range(-1, 1)

    def evaluate(self, val):
        return 2 / (1 + exp(-self.alpha * val)) - 1

    def evaluate_derivative(self, val):
        temp = exp(self.alpha * val)
        return 2 * (self.alpha * temp) / ((temp + 1) ** 2)

    def evaluate_abstract_derivative(self, val):
        raise NotImplementedError


class AdjustableSigmoid(BaseActivation):
    def __init__(self, out_range=Range(-1, 1)):
        super(AdjustableSigmoid, self).__init__()
        self.input_range = Range(float('-inf'), float('inf'))
        self.output_range = out_range

    def evaluate(self, val):
        return self.output_range.delta / (1 + exp(-self.alpha * val)) + self.output_range.min

    def evaluate_derivative(self, val):
        temp = exp(self.alpha * val)
        return self.output_range.delta * (self.alpha * temp) / ((temp + 1) ** 2)

    def evaluate_abstract_derivative(self, val):
        raise NotImplementedError


class HyperbolicTangent(BaseActivation):
    def __init__(self):
        super(HyperbolicTangent, self).__init__()
        self.input_range = Range(float('-inf'), float('inf'))
        self.output_range = Range(-1, 1)

    def evaluate(self, val):
        return tanh(self.alpha * val)

    def evaluate_derivative(self, val):
        return self.alpha * (1 - tanh(self.alpha * val) ** 2)

    def evaluate_abstract_derivative(self, val):
        return self.alpha * (1 - val ** 2)


class Linear(BaseActivation):
    def __init__(self, slope=1):
        super(Linear, self).__init__()
        self.input_range = Range(float('-inf'), float('inf'))
        self.output_range = Range(float('-inf'), float('inf'))
        self.slope = slope

    def evaluate(self, val):
        return self.slope * val

    def evaluate_derivative(self, val):
        return self.slope

    def evaluate_abstract_derivative(self, val):
        raise NotImplementedError
