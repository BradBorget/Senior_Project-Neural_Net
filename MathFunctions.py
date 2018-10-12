import math
import pandas as pd


def oldOutputWeightUpdate(originalWeight, learningRate, actualOutput, expectedOutput, valuePassed): #outdated See new one below
    return originalWeight - (learningRate * (actualOutput - expectedOutput) * valuePassed) #w(ij) = w(ij) - n * (y(j) - t(j)) * x(i)


def NeuronPassFail(listOfWeights, listOfInputs):
    i = 0
    for index in range(len(listOfWeights)):
        if index + 1 - len(listOfInputs) <= 0:
            i += listOfInputs[index] * listOfWeights[index] #This should be weights * inputs
        else:
            i += -1 * listOfWeights[index]
    return 1 / (1 + math.exp(-i))


def weightUpdate(originalWeight, learningRate, errorOutput, input): #Requires Error function first
    return originalWeight - (learningRate * errorOutput * input)


def getOutputError(actualOutput, targetOutput):
    return (actualOutput - targetOutput) * actualOutput * (1 - actualOutput)


def getHiddenError(neuronPassFailValue, listOfWeights, error):
    i = 0
    for index in range(listOfWeights):
        i += error * listOfWeights[index]  # This should be weights * inputs
    return neuronPassFailValue * (1 - neuronPassFailValue) * i


def transfer_derivative(output):
    return output * (1.0 - output)
