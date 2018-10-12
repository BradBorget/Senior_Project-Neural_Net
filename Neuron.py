import random
import pandas as pd
import numpy as np

import MathFunctions as mf


class Neuron:
    def __init__(self, num_weights, letter):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.uniform(-1, 1))
        self.inputs = []
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
    if len(layers) > 1:
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
    output = []
    for layer in network:
        output = []
        for neuron in layer:
            neuron.inputs = inputs
            neuron.inputs.append(-1)
            activate = mf.NeuronPassFail(neuron.weights, neuron.inputs)
            neuron.output = activate
            output.append(neuron.output)
    return output


def back_propagate(layers, target_value, learning_rate):
    output_layer = layers[len(layers)-1]
    for neuron in range(len(output_layer)):
        if neuron == target_value:
            if output_layer[neuron].output <= 0:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 1)
        else:
            if output_layer[neuron].output > 0:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 0)
    for layer in reversed(range(len(layers)-1)):
        for item in range(len(layers[layer])):
            layer[item].Error = mf.getHiddenError(layers[item].output, item, layers)
    for layer in reversed(range(len(layers))):
        for neuron in layers[layer]:
            for weight in range(len(neuron.weights)):
                neuron.weights[weight] = mf.weightUpdate(neuron.weights[weight], learning_rate, neuron.Error, neuron.inputs[weight])




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
    for ninput, row in new_inputs.iterrows():
        while()
        outputs = forward_propagate(neural_network, row.tolist())
        back_propagate(neural_network, row.letter, .1)


if __name__ == "__main__":
    main()

# cycle through each Neuron in the output layer and assign a letter to all 26 Neurons
