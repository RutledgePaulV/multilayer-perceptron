from MultilayerPerceptron.Network import *

def generate(x, y):
    return x**2 + y**2

network = Network(2, [5, 5], 1, 0.5, momentum=0.6180339887)
network.output_layer.activation = AdjustableSigmoid(Range(0,2))

training_set = []

for x in xrange(100):
    in_1, in_2 = random(), random()
    out = generate(in_1, in_2)
    training_set.append(([in_1, in_2], [out]))


for y in xrange(20):
    network.train(training_set, 100)

    in_1, in_2 = random(), random()
    out = generate(in_1, in_2)
    result = network.test([in_1, in_2])

    print "The correct output was: %f" % out
    print "The network output was: %f" % result[0]
    print "The error was: %f" % (out - result[0])
    print "\n"
