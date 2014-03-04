from math import *
from Layer import Layer

class Network(object):
        inputsCount = 0
        layersCount = 0
        output = []
        layers = []
	
        def getLayersCount(self):
                return self.layersCount
        def setLayersCount(self, data):
                self.layersCount = data
        def getLayer(self, index):
                return self.layers[index]
        def getOutput(self):
                return self.output
        def __init__(self, function, inputsCount, neuronsCountPerLayer):
                self.inputsCount = max(1, inputsCount)
                self.layersCount = len(neuronsCountPerLayer) #neuronsCountPerLayer needs to be a list here >.>
                self.layers = [0 for k in range(self.getLayersCount())]
                for i in range(self.getLayersCount()):
                        if i == 0:
                                self.layers[i] = Layer(neuronsCountPerLayer[i], inputsCount, function)
                        else:
                                self.layers[i] = Layer(neuronsCountPerLayer[i], neuronsCountPerLayer[i-1], function)
        #Compute - send a pattern through the network and get the output
        def Compute(self, inputPattern):
                self.output = inputPattern #input needs to be a list! Also, this shouldn't change input at all >.>
                for i in range(self.getLayersCount()):
                        self.output = self.layers[i].Compute(self.output)
                return self.output
	
        #randomize the weights of the ANN
        def Randomize(self):
                for i in range(len(self.layers)):
                        self.layers[i].Randomize()
                return
		
