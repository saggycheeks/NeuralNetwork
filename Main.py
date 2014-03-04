from Layer import Layer
from BackPropogationLearning import BackPropogationLearning
from Neurons import Neuron
from ActivationFunction import *
from Network import Network
from time import *
import cProfile
import csv
import Image
from random import *
import os

#training patterns (Range from -0.5 to 0.5 (white to black)) Letters A - Z
lettera = [-0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5,  0.5, -0.5, 0.5, -0.5,
           -0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, 0.5, -0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

letterb = [0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5]

letterc = [-0.5, -0.5, 0.5, 0.5, 0.5,
           -0.5, 0.5, 0.5, -0.5, -0.5,
           -0.5, 0.5, -0.5, -0.5, -0.5,
           -0.5, 0.5, -0.5, -0.5, -0.5,
           -0.5, 0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, 0.5, 0.5]

letterd = [0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, 0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5]

lettere = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterf = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5]

letterg = [-0.5, -0.5, 0.5, 0.5, 0.5,
           -0.5, 0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, 0.5,
           -0.5, 0.5, -0.5, -0.5, 0.5,
           -0.5, 0.5, 0.5, 0.5, 0.5]

letterh = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

letteri = [0.5, 0.5, 0.5, 0.5, 0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterj = [0.5, 0.5, 0.5, 0.5, 0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5]

letterk = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

letterl = [0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterm = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

lettern = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, 0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

lettero = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterp = [0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5]

letterq = [0.5, 0.5, 0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, 0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5,
           -0.5, -0.5, -0.5, -0.5, 0.5]

letterr = [0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, 0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, 0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, 0.5, -0.5]

letters = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, -0.5]

lettert = [0.5, 0.5, 0.5, 0.5, 0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5]

letteru = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterv = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, 0.5, -0.5, 0.5, 0.5,
           -0.5, 0.5, -0.5, 0.5, -0.5,
           -0.5, 0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5]

letterw = [0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, -0.5, -0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, -0.5, 0.5, -0.5, 0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

letterx = [0.5, -0.5, -0.5, -0.5, 0.5,
           -0.5, 0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, 0.5, -0.5, 0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, 0.5]

lettery = [0.5, -0.5, -0.5, -0.5, 0.5,
           -0.5, 0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5]

letterz = [0.5, 0.5, 0.5, 0.5, 0.5,
           -0.5, -0.5, -0.5, 0.5, -0.5,
           -0.5, -0.5, 0.5, -0.5, -0.5,
           -0.5, 0.5, -0.5, -0.5, -0.5,
           0.5, -0.5, -0.5, -0.5, -0.5,
           0.5, 0.5, 0.5, 0.5, 0.5]

#test pattern storage :/
inputPatterns = [lettera, letterb, letterc, letterd, lettere,
            letterf, letterg, letterh, letteri, letterj,
            letterk, letterl, letterm, lettern, lettero,
            letterp, letterq, letterr, letters, lettert,
            letteru, letterv, letterw, letterx, lettery, letterz]

#Output patterns (0.5 (100% match) at the index corresponding to the letter. 0 for A, 1 for B, etc. -0.5 (0% match) everywhere else.
outputA = [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputB = [-0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputC = [-0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputD = [-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputE = [-0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputF = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputG = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputH = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputI = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputJ = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputK = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputL = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputM = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputN = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputO = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputP = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputQ = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputR = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputS = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputT = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputU = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
outputV = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5]
outputW = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5]
outputX = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5]
outputY = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5]
outputZ = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]

#output pattern storage
outputPatterns = [outputA, outputB, outputC, outputD, outputE, outputF, outputG, outputH, outputI,
                  outputJ, outputK, outputL, outputM, outputN, outputO, outputP, outputQ, outputR,
                  outputS, outputT, outputU, outputV, outputW, outputX, outputY, outputZ]

patternSize = 30
patterns = [26]

#Save the weights in a 2d list and write them to a csv file
def SaveWeights(neuralNet, filename):
    #write the weights for each node to a file to save them for later o.O
    savedNeurons = []
    for i in range(neuralNet.getLayer(0).getNeuronsCount()):
        savedWeights = []
        for x in range(neuralNet.getLayer(0).getNeuron(i).getInputsCount()):
            savedWeights.append(neuralNet.getLayer(0).getNeuron(i).getWeight(x))

        savedNeurons.append(savedWeights)
  
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter =' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(savedNeurons)):
            writer.writerow(savedNeurons[i])

    print "{0} written successfully. (Hopefully)".format(filename)  

#This assumes only a single layer. I won't need anything more.
def LoadWeights(network, filename):
    loadedWeights = None
    loadedNeurons = []

    #read the weights in from the csv file
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            loadedWeights = []
            for i in range(len(row)):
                loadedWeights.append(float(row[i]))
            loadedNeurons.append(loadedWeights)

    #save the data to the network
    for i in range(network.getLayer(0).getNeuronsCount()):
        for k in range(network.getLayer(0).getNeuron(i).getInputsCount()):
            network.getLayer(0).getNeuron(i).setWeight(k, loadedNeurons[i][k])
    return

def TeachNetwork(neuralNet):
    #make the teacher
    teacher = BackPropogationLearning(neuralNet)
    teacher.setLearningLimit(0.1)
    teacher.setLearningRate(0.5)

    #teach the network
    i = 0
    startTime = clock()
    while (True):
        error = teacher.LearnEpoch(inputPatterns, outputPatterns)
        i += 1
        if(teacher.getConverged()):
            break
    endTime = clock()
    print "Learning time: {0} Minutes".format((endTime - startTime)/60)
    print "Total learning epoch: {0}".format(i)
    print "error: {0}".format(error)
    return

def CreateImage(pattern, filename):
    #DO NOT PROVIDE THE REFERENCE TO THE ORIGINAL PATTERN
    #USE list(pattern) WHEN PASSING THE PATTERN
    #filename MUST HAVE AN EXTENSION!
    #invert the pattern (just multiply by -1. in this case, it'll work)
    for i in range(len(pattern)):
        pattern[i] *= -1

    #convert the pattern data from [-0.5, 0.5] to [0, 255]
    for i in range(len(pattern)):
        pattern[i] = int((pattern[i] + 0.5) * 255)

    #Convert the list from single values to three values per entry
    #Also, each entry should be a tuple.
    for i in range(len(pattern)):
        x = pattern[i]
        pattern[i] = (x, x, x)

    #create a tuple from the list
    data = tuple(pattern)
    #create an image (RGB format, 5 x 6)
    image = Image.new('RGB', (5, 6))
    image.putdata(pattern)
    #resize the image for better viewing
    image2 = image.resize((256, 300))
    #save the image
    image2.save(filename)
    return


def SavePatternToCSV(pattern, filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar = '|')
        writer.writerow(pattern)
    return

def LoadPatternFromCSV(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar = '|')
        pattern = reader.next()
        #convert the input from strings to floats
        for i in range(len(pattern)):
            pattern[i] = float(pattern[i])
    return pattern

#returns a list of guesses for the supplied patterns
#the index of the guess corresponds to the index of the pattern
#the value at that index is the guessed letter. Range: [0, 25] 0 is A, 25 is Z
def GuessPatterns(neuralNet, inputPatterns):
    highestMatch = 0.0
    
    output = []
    for inputIndex in range(len(inputPatterns)):
        bestGuess = 0
        guess = neuralNet.Compute(inputPatterns[inputIndex])
        highestMatch = guess[0]
        for outputIndex in range(len(guess)):
            if guess[outputIndex] > highestMatch:
                highestMatch = guess[outputIndex]
                bestGuess = outputIndex
        output.append(bestGuess)
    return output

def InsertNoise(pattern, numPixels):
    if numPixels >= (patternSize / 2):
        answer = raw_input("Sanity check: Are you insane? ")
        if answer == "no":
            print "Okay. You're the boss."
        else:
            print "Yeah. Maybe you should lie down for a bit."
            return
        
    for i in range(numPixels):
        index = randint(0, 29)
        noise = random() - 0.5
        pattern[index] = noise

def AppendFolder(filename):
    p1 = os.path.join("data", "presentation")
    path = os.path.join(p1, filename)
    return path

#create neural network, Bipolar Sigmoid Function with alpha = 1.0
neuralNet = Network(BipolarSigmoidFunction(1.0), patternSize, patterns)
#Randomize the weights
#Needs to be done before loading the weights from a file
#otherwise the lists aren't created yet.
neuralNet.Randomize()
LoadWeights(neuralNet, "weights.csv")

##tempPattern = list(inputPatterns[20])
##
##output = neuralNet.Compute(tempPattern)
##
##for i in range(len(output)):
##    print "{0}: {1}".format(chr(ord('A') + i), 100 * (output[i] + 0.5))

pattern = []

while(True):
    print "*********************************"
    print "**********Main Menu**************"
    print "*********************************"
    print "\n(0) Quit"
    print "(1) Run test patterns through the neural network"
    print "(2) Run a single pattern through the neural network"
    print "(3) Load a pattern from disk"
    print "(4) Save a pattern to disk"
    print "(5) Save an image of a pattern to disk"
    print "(6) Insert noise into a pattern"
    print "(7) See match percentage for Pattern"
    choice = raw_input("Choice: ")
    choice = int(choice)
    print "\n\n"

    if choice == 0:
        raise SystemExit
        
    if choice == 1:
        output = GuessPatterns(neuralNet, inputPatterns)
        for i in range(len(output)):
            print "Letter is {0} and NeuralNetwork thinks it is {1}.".format(chr(ord('A') + i), chr(ord('A') + output[i]))
        continue

    elif choice == 3:
        
        filename = raw_input("filename: ")
        filename = AppendFolder(filename)
        pattern = LoadPatternFromCSV(filename)
        print "Pattern Loaded Successfully."

    elif choice == 2:
        choice = raw_input("(T)est pattern or (L)oaded pattern? ")
        if choice == "T" or choice == "t":
            choice = raw_input("Which one? (A - Z)")
            choice = ord(choice) - ord('A')
            pattern = list(inputPatterns[choice])
        elif choice == 'l' or choice == 'L':
            if len(pattern) == 0:
                print "No pattern loaded! Aborting."
                print "\n\n"
                continue
        output = GuessPatterns(neuralNet, [pattern])
        print "Neural Network thinks this is the letter {0}.".format(chr(ord('A') + output[0]))

    elif choice == 4:
        filename = raw_input("filename: ")
        filename = AppendFolder(filename)
        SavePatternToCSV(pattern, filename)
        print "file \"{0}\" saved successfully.".format(filename)

    elif choice == 5:
        filename = raw_input("filename: ")
        filename = AppendFolder(filename)
        CreateImage(list(pattern), filename)
        print "file \"{0}\" saved successfully.".format(filename)

    elif choice == 6:
        choice = raw_input("(T)est pattern or (L)oaded pattern? ")
        if choice == "T" or choice == "t":
            choice = raw_input("Which one? (A - Z)")
            choice = ord(choice) - ord('A')
            pattern = list(inputPatterns[choice])
        elif choice == 'l' or choice == 'L':
            if len(pattern) == 0:
                print "No pattern loaded! Aborting."
                print "\n\n"
                continue
        numPixels = int(raw_input("How many pixels to randomize? "))
        InsertNoise(pattern, numPixels)
        print "Done."

    elif choice == 7:
        output = neuralNet.Compute(pattern)
        for i in range(len(output)):
            print "{0}: {1}".format(chr(ord('A') + i), 100 * (output[i] + 0.5))

    else:
        print "Boss, that made no sense. Try again."

    print "\n\n"
    continue
        
        
##result = GuessPatterns(neuralNet, inputPatterns)
##for i in range(len(result)):
##    print "The test letter is {0} and the network thinks it's {1}".format(chr(ord('A') + i), chr(ord('A') + result[i]))

