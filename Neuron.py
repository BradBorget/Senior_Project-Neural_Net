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
    new_inputs = inputs
    for layer in network:
        for neuron in layer:
            neuron.inputs = new_inputs
            neuron.inputs.append(-1)
            activate = mf.NeuronPassFail(neuron.weights, neuron.inputs)
            neuron.output = activate
            #output.append(neuron.output)
        #new_inputs = output


def back_propagate(layers, target_value, learning_rate):
    output_layer = layers[len(layers)-1]
    for neuron in range(len(output_layer)):
        if neuron == target_value:
            if output_layer[neuron].output <= .5:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 1)
        else:
            if output_layer[neuron].output > .5:
                output_layer[neuron].Error = mf.getOutputError(output_layer[neuron].output, 0)
    for layer in reversed(range(len(layers)-2)):
        for item in range(len(layers[layer])):
            layers[layer][item].Error = mf.getHiddenError(layer, item, layers)
    for layer in range(len(layers)):
        for neuron in layers[layer]:
            for weight in range(len(neuron.weights)):
                neuron.weights[weight] = mf.weightUpdate(neuron.weights[weight], learning_rate, neuron.Error, neuron.inputs[weight])


# change targets to numbers using dictionary
def evaluate_algorithm(dataset, *args):
    seed = random.randint(0, 999)

    train_set, test_set, targets_train, targets_test = \
        train_test_split(dataset.loc[:, dataset.columns != 'letter'].values, dataset.letter.values, test_size=0.30, random_state=seed)
    #folds = cross_validation_split(dataset, n_folds)
    #scores = list()
    letters_dic = setup_letters(dataset.letter)
    #for fold in folds:
     #   train_set = list(folds)
      #  train_set.remove(fold)
       # train_set = sum(train_set, [])
        #test_set = list()
        #for row in fold:
         #   row_copy = list(row)
          ##  test_set.append(row_copy)
            #row_copy[-1] = None
    targets = []
    test_targets = []
    for values in targets_train:
        targets.append(int(letters_dic[values]))
    for values in targets_test:
        test_targets.append(int(letters_dic[values]))
    accuracy = propagation(train_set, test_set, targets,  *args,test_targets)

    return accuracy


#
def propagation(train, test, targets_train, l_rate, n_epoch, n_hidden, test_targets):
    n_inputs = len(train[1])
    n_outputs = len(set(targets_train))

    network = create_network(n_hidden, n_outputs, n_inputs)
    for epoch in range(n_epoch):
        train_network(network, train, targets_train, l_rate, n_epoch)
        predictions = list()
        output_layer = (network[len(network) -1])
        for row in test:
            prediction = list()
            forward_propagate(network, list(row))
            for neuron in range(len(output_layer)):
                if output_layer[neuron].output > .5:
                    prediction.append(neuron)
            predictions.append(prediction)
        accuracy = accuracy_metric(test_targets, predictions)
        print('Scores: %s' % accuracy)
    return(accuracy)


#Wwork on verifying results
def train_network(network, train, targets_train, l_rate, n_epoch):
    n = 0
    for row in range(len(train)):
        i = 0
        output_layer = network[-1]
        while output_layer[targets_train[row]].output <= .5:
            forward_propagate(network, list(train[row]))
            back_propagate(network, targets_train[row], l_rate)
            i += 1
            if(i % 50 == 0):
                print("break")
        n += 1
        if 0 == n % 2000:
            print(n)


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        for j in predicted[i]:
            if actual[i] == j:
                correct += 1
    return correct / float(len(actual)) * 100.0


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)

    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


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


def load_data():
    inputs = pd.read_csv("letter-recognition.txt")
    letters = inputs.letter
    new_inputs = inputs.loc[:, inputs.columns != 'letter']
    return letters, new_inputs


def set_up_hidden_layers():
    num_layers = int(input("Enter number of hidden layers: "))
    layers = []
    for i in range(0, num_layers):
        layer = int(input("Enter number of Neurons for hidden layer " + str(i+1) + ": "))
        layers.append(layer)
    return layers
    neural_network = create_network(layers, output_layer, len(new_inputs.columns))
    for ninput, row in new_inputs.iterrows():
        for i in range(10):
            forward_propagate(neural_network, row.tolist())
            back_propagate(neural_network, letters_dic[letters[ninput]], .1)
    print(neural_network)

def main():
    #letters, new_inputs = load_data()
    dataset = pd.read_csv("letter-recognition.txt")
    output_layer = len(set(dataset))
    #letters_dic = setup_letters(letters)
    layers = set_up_hidden_layers()
    #n_folds = 5
    l_rate = 0.3
    n_epoch = 50

    scores = evaluate_algorithm(dataset, l_rate, n_epoch, layers)
    print('Scores: %s' % scores)
    #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

    #neural_network = create_network(layers, output_layer, len(new_inputs.columns))
    #for ninput, row in new_inputs.iterrows():
      #  forward_propagate(neural_network, row.tolist())
      #  back_propagate(neural_network, letters_dic[letters[ninput]], .1)


if __name__ == "__main__":
    main()


# cycle through each Neuron in the output layer and assign a letter to all 26 Neurons
