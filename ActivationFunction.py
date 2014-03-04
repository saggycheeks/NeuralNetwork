"""
class IActivationFunction(Object):
	def Output(self, input):
	def OutputPrime(self, input):
	def OutputPrime2(self, input):
"""

from math import *

class SigmoidFunction(object):
	alpha = 2.0
	def getAlpha(self):
		return self.alpha
	def setAlpha(self, input):
		self.alpha = input
	#Because apparently you can't overload __init__, so always supply *something*"
	def __init__(self, input):
		self.alpha = input

	def Output(self, x):
		return (1.0 / (1.0 + exp(-self.alpha * x)))

	def OutputPrime(self, x):
		y = self.Output(x)
		return (self.alpha * y * (1.0 - y))
	
	def OutputPrime2(self, y):
		return (self.alpha * y * (1.0 - y))

class BipolarSigmoidFunction(object):
	alpha = 2.0
	def getAlpha(self):
		return self.alpha
	#this might be an issue if data isn't a float! Just check it in the main program.
	def setAlpha(self, data):
		self.alpha = data
	def __init__(self, data):
		self.alpha = data
	def Output(self, x):
		return ((1.0 / (1.0 + exp(-(self.alpha) * x))) - 0.5)
	def OutputPrime(self, x):
		y = self.Output(x)
		return (self.alpha * (0.25 - y * y))
	def OutputPrime2(self, y):
		return (self.alpha * (0.25 - y * y))

class HyperbolicTangensFunction(object):
	alpha = 1.0
	def setAlpha(self, data):
		self.alpha = data
	def getAlpha(self):
		return self.alpha
	def __init__(self, data):
		self.alpha = data
	def Output(self, x):
		return (tanh(self.alpha * x))
	def OutputPrime(self, x):
		y = self.Output(x)
		return (self.alpha * (1 - y * y))
	def OutputPrime2(self, y):
		return (self.alpha * (1 - y * y))
	
	
