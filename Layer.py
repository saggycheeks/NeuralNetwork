from Neurons import Neuron

#layers hold the neurons
#This is basically a wrapper class
class Layer(object):
	inputsCount = 0
	neuronsCount = 0
	function = None
	neurons = []
	output = []
	
	def getInputsCount(self):
		return self.inputsCount
	def setInputsCount(self, data):
		self.inputsCount = data
	def getNeuronsCount(self):
		return self.neuronsCount
	def setNeuronsCount(self, data):
		self.neuronsCount = data
		self.initLayer()
	def getFunction(self):
		return self.function
	def setFunction(self, value):
		self.function = value
		for i in range(self.neuronsCount):
			self.neurons[i].setFunction(value)
	def getNeuron(self, index):
		return self.neurons[index]
	def getOutput(self):
		return self.output
	def __init__(self, neuronsCount, inputsCount, function):
		self.neuronsCount = neuronsCount
		self.inputsCount = inputsCount
		self.function = function
		self.InitLayer()

	def Compute(self, inputPattern):
		for i in range(self.neuronsCount):
			self.output[i] = self.neurons[i].Compute(inputPattern)
		return self.output
	def Randomize(self):
		for x in range(len(self.neurons)):
			self.neurons[x].Randomize()

	def InitLayer(self):
		self.neurons = [0 for i in range(self.neuronsCount)]
                #neurons = []
		for i in range(self.neuronsCount):
			self.neurons[i] = (Neuron(self.inputsCount, self.function))
			self.output.append(0.0)
	
	
