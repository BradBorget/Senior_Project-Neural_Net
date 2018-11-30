import random
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import MathFunctions as mf
from random import seed
from random import randrange
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix


class Neuron:
    def __init__(self, num_weights):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.random())
        self.inputs = []
        self.output = 0
        self.Error = 0



def create_network(n_hidden, output_layer, num_weight):
    layer_list = []
    layer_list.append(create_layer(num_weight, n_hidden))
    layer_list.append(create_layer(n_hidden, output_layer))
    return layer_list


def create_layer(num_inputs, num_neurons):
    neuron_list = []
    for i in range(0, num_neurons):
        neuron_list.append(Neuron(num_inputs))
    return neuron_list


def forward_propagate(network, inputs):
    new_inputs = inputs
    for layer in network:
        output = []
        for neuron in layer:
            neuron.inputs = new_inputs
            activate = mf.NeuronPassFail(neuron.weights, neuron.inputs)
            neuron.output = activate
            output.append(neuron.output)
        new_inputs = output



def back_propagate(layers, target_value, learning_rate, row):
    output_layer = layers[1]
    for index in range(len(output_layer)):
        if index == target_value:
            if output_layer[index].output <= .5:
                output_layer[index].Error = mf.getOutputError(output_layer[index].output, 1)
        else:
            if output_layer[index].output > .5:
                output_layer[index].Error = mf.getOutputError(output_layer[index].output, 0)
    for layer in reversed(range(len(layers)-2)):
        for item in range(len(layers[layer])):
            layers[layer][item].Error = mf.getHiddenError(layer, item, layers)
    for layer in range(len(layers)):
        inputs = row[:-1]
        if layer != 0:
            inputs = [neuron.output for neuron in layers[layer - 1]]
        for neuron in layers[layer]:
            for j in range(len(inputs)):
                neuron.weights[j] = mf.weightUpdate(neuron.weights[j],
                                                         learning_rate, neuron.Error, inputs[j])



            #for weight in range(len(neuron.weights)):
                #neuron.weights[weight] = mf.weightUpdate(neuron.weights[weight], learning_rate, neuron.Error, neuron.inputs[weight])


# change targets to numbers using dictionary
def evaluate_algorithm(dataset, *args):
    seed = random.randint(0, 999)
    minmax = dataset_minmax(dataset.loc[:, dataset.columns != 'letter'].values)
    normalize_dataset(dataset.loc[:, dataset.columns != 'letter'].values, minmax)
    train_set, test_set, targets_train, targets_test = \
        train_test_split(dataset.loc[:, dataset.columns != 'letter'].values, dataset.letter.values, test_size=0.30, random_state=seed)
    letters_dic = setup_letters(dataset.letter)
    targets = []
    test_targets = []
    for values in targets_train:
        targets.append(int(letters_dic[values]))
    for values in targets_test:
        test_targets.append(int(letters_dic[values]))
    #for values in targets_train:
     #   targets.append(values)
    #for values in targets_test:
     #   test_targets.append(values)

    accuracy = propagation(train_set, test_set, targets,  *args,test_targets)

    return accuracy


#
def propagation(train, test, targets_train, l_rate, n_epoch, n_hidden, test_targets):
    n_inputs = len(train[0])
    n_outputs = len(set(targets_train))
    accuracy = 0
    network = create_network(n_hidden, n_outputs, n_inputs)
    for epoch in range(n_epoch):
        train_network(network, train, targets_train, l_rate)
        predictions = list()
        output_layer = (network[len(network) -1])
        for row in test:
            prediction = list()
            forward_propagate(network, list(row))
            for neuron in range(len(output_layer)):
                if output_layer[neuron].output > .5:
                    prediction.append(neuron)
            predictions.append(prediction)
        predicted_letters = []
        actual_letters = []
        p_letter_array = []
        a_letter_array = []
        labels = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
                  "V", "W", "X", "Y", "Z"]
        for row in range(len(prediction)):
            p_letter_array.append(prediction[row])
        letter_dict = put_letters(p_letter_array)
        for row in range(len(prediction)):
            predicted_letters.append(letter_dict[p_letter_array[row]])
        for row in range(len(test_targets)):
            a_letter_array.append(test_targets[row])
        letter_dict = put_letters(a_letter_array)
        for row in range(len(test_targets)):
            actual_letters.append(letter_dict[a_letter_array[row]])
        cm = confusion_matrix(actual_letters, predicted_letters, labels)
        print_cm(cm, labels)
        accuracy = accuracy_metric(test_targets, predictions)
        print('Scores: %s' % accuracy)
    return accuracy


#Wwork on verifying results
def train_network(network, train, targets_train, l_rate):
    n = 0
    for row in range(len(train)):
        output_layer = network[-1]
        forward_propagate(network, list(train[row]))
        back_propagate(network, targets_train[row], l_rate, list(train[row]))
        n += 1
        if 0 == n % 2000:
            print(n)


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        print(str(actual[i]) + "        " + str(predicted[i]))
        for j in predicted[i]:
            if actual[i] == j:
                correct += 1
    return correct / float(len(actual)) * 100.0


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


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


def put_letters(letters_input):
    letters = dict.fromkeys(letters_input)
    letters[0] = 'A'
    letters[1] = 'B'
    letters[2] = 'C'
    letters[3] = 'D'
    letters[4] = 'E'
    letters[5] = 'F'
    letters[6] = 'G'
    letters[7] = 'H'
    letters[8] = 'I'
    letters[9] = 'J'
    letters[10] = 'K'
    letters[11] = 'L'
    letters[12] = 'M'
    letters[13] = 'N'
    letters[14] = 'O'
    letters[15] = 'P'
    letters[16] = 'Q'
    letters[17] = 'R'
    letters[18] = 'S'
    letters[19] = 'T'
    letters[20] = 'U'
    letters[21] = 'V'
    letters[22] = 'W'
    letters[23] = 'X'
    letters[24] = 'Y'
    letters[25] = 'Z'
    return letters


def create_network(layers, output_layer, num_weight):
    layer_list = []
    layer_list.append(create_layer(num_weight, layers[0]))
    j = 0
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            layer_list.append(create_layer(layers[i], layers[(i + 1)]))
            j = i
    layer_list.append(create_layer(layers[j], output_layer))
    return layer_list


def set_up_hidden_layers():
    num_layers = int(input("Enter number of hidden layers: "))
    layers = []
    for i in range(0, num_layers):
        layer = int(input("Enter number of Neurons for hidden layer " + str(i+1) + ": "))
        layers.append(layer)
    return layers




def main():
    dataset = pd.read_csv("letter-recognition.txt")
    output_layer = len(set(dataset))
    #letters_dic = setup_letters(letters)
    layers = set_up_hidden_layers()
    l_rate = 0.3
    n_epoch = 50
    scores = evaluate_algorithm(dataset, l_rate, n_epoch, layers)
    print('Scores: %s' % scores)


if __name__ == "__main__":
    main()


# cycle through each Neuron in the output layer and assign a letter to all 26 Neurons
