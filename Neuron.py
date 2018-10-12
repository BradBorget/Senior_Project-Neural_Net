import random
import pandas as pd
import numpy as np

import MathFunctions as mf


class Neuron:
    def __init__(self, num_weights, letter):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.uniform(-1, 1))
        #self.inputs = inputs
        #self.Bj = 0 #I can't remember what B sub j represents, so I'm commenting it out for now.  Putting in Error instead to represent the value that the neuron came to
        self.output = 0
        self.Error = 0
        self.letter = letter


class NeuralNetClass:
    def __init__(self):
        pass

    def fit(self, data, target):
        pass


class NeuralNetModel:
    def __init__(self):
        pass

    def predict(self, data):
        pass


def create_network(layers, output_layer, num_weight):
    layer_list = []
    layer_list.append(create_layer(num_weight, layers[0]))
    j = 0
    if(len(layers) > 1):
        for i in range(len(layers)-1):
            layer_list.append(create_layer(layers[i], layers[(i+1)]))
            j = i
    layer_list.append(create_layer(layers[j], output_layer))
    return layer_list


def create_layer(num_inputs, num_neurons):
    neuron_list = []
    for i in range(0, num_neurons):
        neuron_list.append(Neuron(num_inputs, ""))
    return neuron_list


def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activate = mf.NeuronPassFail(neuron.weights, inputs)
            neuron.output = activate
            new_inputs.append(neuron.output)
    return new_inputs


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron.weights[j] * neuron.error)
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron.output)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron.error = errors[j] * mf.transfer_derivative(neuron.output)


def main():
    inputs = pd.read_csv("letter-recognition.txt")
    output_layer = len(set(inputs.letter))
    letters = inputs.letter
    new_inputs = inputs.loc[:, inputs.columns != 'letter']
    num_layers = int(input("Enter number of hidden layers: "))
    layers = []
    for i in range(0, num_layers):
        layer = int(input("Enter number of Neurons for hidden layer " + str(i+1) + ": "))
        layers.append(layer)
    neural_network = create_network(layers, output_layer, len(new_inputs.columns))
    outputs = forward_propagate(neural_network, new_inputs)
    backward_propagate_error(neural_network, letters)
    print(neural_network)


if __name__ == "__main__":
    main()

# cycle through each Neuron in the output layer and assign a letter to all 26 Neurons
