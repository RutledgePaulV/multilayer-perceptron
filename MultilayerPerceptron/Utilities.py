import abc
from random import *

class Range(object):
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def is_in_range(self, val):
        return not self.min > val < self.max

    @property
    def delta(self):
        return self.max - self.min


class BaseRandom(object):
    def __init__(self, output_range=Range(0.0, 1.0)):
        self.output_range = output_range

    @abc.abstractmethod
    def generate(self):
        raise NotImplementedError


class Standard(BaseRandom):
    def __init__(self, output_range=Range(0.0, 1.0)):
        super(Standard, self).__init__(output_range)

    def generate(self):
        return random() * self.output_range.delta + self.output_range.min