import random
import pandas as pd
import numpy as np
import itertools

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
    for layer in network:
        for neuron in layer:
            neuron.inputs = inputs
            neuron.inputs.append(-1)
            activate = mf.NeuronPassFail(neuron.weights, neuron.inputs)
            neuron.output = activate


def back_propagate(layers, target_value, learning_rate):
    output_layer = layers[len(layers)-1]
    for neuron in range(len(output_layer)):
        if neuron == target_value:
            if output_layer[neuron].output <= 0:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 1)
        else:
            if output_layer[neuron].output > 0:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 0)
    for layer in reversed(range(len(layers)-2)):
        for item in range(len(layers[layer])):
            layers[layer][item].Error = mf.getHiddenError(layer, item, layers)
    for layer in range(len(layers)):
        for neuron in layers[layer]:
            for weight in range(len(neuron.weights)):
                neuron.weights[weight] = mf.weightUpdate(neuron.weights[weight], learning_rate, neuron.Error,
                                                         neuron.inputs[weight])


def setup_letters(letters_input):
    letters = dict.fromkeys(letters_input.tolist())
    letters['A'] = 0
    letters['B'] = 1
    letters['C'] = 2
    letters['D'] = 3
    letters['E'] = 4
    letters['F'] = 5
    letters['G'] = 6
    letters['H'] = 7
    letters['I'] = 8
    letters['J'] = 9
    letters['K'] = 10
    letters['L'] = 11
    letters['M'] = 12
    letters['N'] = 13
    letters['O'] = 14
    letters['P'] = 15
    letters['Q'] = 16
    letters['R'] = 17
    letters['S'] = 18
    letters['T'] = 19
    letters['U'] = 20
    letters['V'] = 21
    letters['W'] = 22
    letters['X'] = 23
    letters['Y'] = 24
    letters['Z'] = 25
    return letters


def main():
    inputs = pd.read_csv("C:\\Users\\bradl\\Documents\\GitHub\\Senior_Project-Neural_Net\\letter-recognition.txt")
    output_layer = len(set(inputs.letter))
    letters = inputs.letter

    letters_dic = setup_letters(letters)
    new_inputs = inputs.loc[:, inputs.columns != 'letter']

    num_layers = int(input("Enter number of hidden layers: "))
    layers = []
    for i in range(0, num_layers):
        layer = int(input("Enter number of Neurons for hidden layer " + str(i+1) + ": "))
        layers.append(layer)

    neural_network = create_network(layers, output_layer, len(new_inputs.columns))
    for ninput, row in new_inputs.iterrows():
        for i in range(10):
            outputs = forward_propagate(neural_network, row.tolist())
            back_propagate(neural_network, letters_dic[letters[ninput]], .1)
    print(neural_network)


if __name__ == "__main__":
    main()

# cycle through each Neuron in the output layer and assign a letter to all 26 Neurons
