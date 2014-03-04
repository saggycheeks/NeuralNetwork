
class BackPropogationLearning(object):
        net = None
        learningRate = 0.1
        momentum = 0.0
        learningLimit = 0.1

        converged = False
	
        #Not multi-dimensional --- its an array of an array of arrays
        #Implement as lists. it' staggered anyway
	
        errors = []
        deltas = []
        thresholdDeltas = []

        def getLearningRate(self):
                return self.learningRate
        def setLearningRate(self, data):
                self.learningRate = data
        def setLearningLimit(self, data):
                self.lerningLimit = data
        def getLearningLimit(self):
                return self.learningLimit
        def getMomentum(self):
                return self.momentum
        def setMomentum(self, data):
                self.momentum = data
        def getConverged(self):
                return self.converged
        def setConverged(self, data):
                #set a check for boolean type here, then throw error...
                self.converged = data
        def __init__(self, net):
                self.net = net
                #init these lists and make them the right size
                self.errors = [[] for i in range(net.getLayersCount())]
                self.deltas = [[] for i in range(net.getLayersCount())]
                self.thresholdDeltas = [[] for i in range(net.getLayersCount())]

                #initialize each entry in deltas to list of size layer.getInputsCount()
                for i in range(self.net.getLayersCount()):
                        layer = self.net.getLayer(i)
                        #create lists of size NeuronsCount in the current selected list
                        self.errors[i] = [0.0 for x in range(layer.getNeuronsCount())]
                        self.deltas[i] = [[] for x in range(layer.getNeuronsCount())]
                        self.thresholdDeltas[i] = [0.0 for x in range(layer.getNeuronsCount())]
                        for j in range(layer.getNeuronsCount()):
                                self.deltas[i][j] = [0.0 for x in range(layer.getInputsCount())]
                return
	#main learning function. This should be called once every iteration needed.
        def LearnEpoch(self, inputPatterns, output):
                i = None
                n = len(inputPatterns)
                error = 0.0
		
                for x in range(n):
                        error += self.Learn(inputPatterns[x], output[x])
                self.converged = (error < self.learningLimit)
                return error
        #called once for every pattern.
        #calculate the error for the pattern and update the network
        def Learn(self, inputPatterns, output):
                nout = self.net.Compute(inputPatterns)
                error = self.CalculateError(output)
                self.CalculateUpdates(inputPatterns)
                self.UpdateNetwork()
                return error	

        def CalculateError(self, desiredOutput):
                layer = None
                layerNext = None
                err = []
                errNext = []
                error = 0.0
                e = None
                output = 0.0
                mySum = 0.0
                layersCount = self.net.getLayersCount()

                #fuck ambiguous code. the guy that did this in c# is a fucking jackass.
                #setup for determining error
                function = self.net.getLayer(0).getNeuron(0).getFunction()
                layer = self.net.getLayer(layersCount - 1)
                err = self.errors[layersCount - 1]
                #get error for every neuron
                for i in range(layer.getNeuronsCount()):
                        output = layer.getNeuron(i).getOutput()
                        e = desiredOutput[i] - output
                        err[i] = e * function.OutputPrime2(output)
                        error += (e * e)

                #bottom part just does nothing if layersCount == 1...

                temp = range(layersCount - 2)
                temp.reverse()
                for j in temp:
                        layer = net.getLayer(j)
                        layerNext = net.getLayer(j+1)
                        err = errors[j]
                        errNext = errors[j + 1]
                        for i in range(layer.getNeuronsCount()):
                                mySum = 0.0
                                for k in range(layerNext.getNeuronsCount()):
                                        mySum += errNext[k] * layerNext.getNeuron(k).getWeight(i)
                                err[i] = mySum * function.OutputPrime2( layer.getNueron(i).Output)

                return error

        def CalculateUpdates(self, input):
                neuron = None
                layer = None
                layerPrev = None
                lDeltas = []  #An array of arrays... not exactly 2 dimensional but almost
                err = []
                delt = []
                tdel = []
                e = 0.0

                #for first layer. . . 
                layer = self.net.getLayer(0)
                lDeltas = self.deltas[0]
                err = self.errors[0]
                tdel = self.thresholdDeltas[0]

                #for each neuron of the layer
                for i in range(layer.getNeuronsCount()):
                        neuron = layer.getNeuron(i)
                        delt = lDeltas[i]
                        e = err[i]

                        #for each synapse of the neuron
                        for j in range(neuron.getInputsCount()):
                                #Calculate synapse update
                                delt[j] = self.learningRate * (self.momentum * delt[j] + (1.0 - self.momentum) * e * input[j])
                        #calculate threshold update
                        tdel[i] = self.learningRate * (self.momentum * tdel[i] + (1.0 - self.momentum) * e)


                #for all other layers (1 to LayersCount - 1)
                #This doesn't matter for ANN with only 1 layer
                for k in range(1, self.net.getLayersCount()):
                        layerPrev = self.net.getLayer(k - 1)
                        layer = self.net.getLayer(k)
                        lDeltas = self.deltas[k]
                        err = self.errors[k]
                        tdel = self.thresholdDeltas[k]

                        for i in range(layer.getNeuronsCount()):
                                neuron = layer.getNeuron(i)
                                delt = lDeltas[i]
                                e = err[i]
                                #for each synapse of the neuron
                                for j in range(neuron.getInputsCount()):
                                        #calculate weight update
                                        delt[j] = self.learningRate * (self.momentum * delt[j] + (1.0 - self.momentum) * e * layerPrev.getNeuron[j].getOutput())
                                #calculate threshold update
                                tdel[i] = self.learningRate * (self.momentum * tdel[i] + (1.0 - self.momentum) * e)

        def UpdateNetwork(self):
                neuron = None
                layer = None
                lDeltas = []
                delt = []
                tdel = []

                #for each layer in the network
                for i in range(self.net.getLayersCount()):
                        layer = self.net.getLayer(i)
                        lDeltas = self.deltas[i]
                        tdel = self.thresholdDeltas[i]

                        #for each neuron in the layyer
                        for j in range(layer.getNeuronsCount()):
                                neuron = layer.getNeuron(j)
                                delt = lDeltas[j]
                                #for each weight of the neuron
                                for k in range(neuron.getInputsCount()):
                                        #update the weights for the neurons
                                        neuron.setWeight(k, neuron.getWeight(k) + delt[k])
                                #update threshold
                                neuron.setThreshold(neuron.getThreshold() - tdel[j])
                                
                                
