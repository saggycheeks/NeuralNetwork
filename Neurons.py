from ActivationFunction import *
from random import *

class Neuron(object):
        inputsCount = 1
        weights = []
        threshold = 0.0
        function = SigmoidFunction(2)
        mySum = 0.0
        output = 0.0
	
        def setInputsCount(self, data):
                self.inputsCount = max(1, data)
                #resize weights to the right size with dummy data
                self.weights = []
                for i in range(self.inputsCount):
                        self.weights.append(0.0)
		
        def getInputsCount(self):
                return self.inputsCount
	
        def setThreshold(self, data):
                self.threshold = data
        def getThreshold(self):
                return self.threshold

        def setFunction(self, data):
                self.function = data
        def getFunction(self):
                return self.function
	
        def getOutput(self):
                return self.output
	
        def setWeight(self, index, value):
                self.weights[index] = value
        def getWeight(self, index):
                return self.weights[index]

        def __init__(self, inputs, function):
                self.setFunction(function)
                self.setInputsCount(inputs)

        #Take the sum of the product of the synapses with their respective pixels
        #and send them through the activation function. Send the result back as
        #the output.
        def Compute(self, inputPattern):
                if(len(inputPattern) != self.getInputsCount()):
                        print "Error in Neurons.py : Compute() - bad input length. (Should be a list of length {0} )".format(self.getInputsCount())
                        return
                self.mySum = 0.0
		
                for i in range(self.inputsCount):
                        self.mySum += self.weights[i] * inputPattern[i]
		
                self.mySum -= self.threshold
                #interpretation here due to ambiguous code... again...
                self.output = self.function.Output(self.mySum)
                return self.output

        def Randomize(self):
                seed()
                for i in range(self.inputsCount):
                        self.weights[i] = random()
                self.threshold = random()

