import random
import pandas as pd

#import MathFunctions as mf


class Neuron:
    def __init__(self, num_weights, letter=""):
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
    j = 0
    layer_list.append(create_layer(num_weight, layers[j]))
    for i in layers[-0]:
        layer_list.append(create_layer(layers[j], i))
        j += 1
    layer_list.append(create_layer(layers[j], output_layer))
    return layer_list


def create_layer(num_inputs, numNeurons):
    list = []
    for i in range(0, numNeurons):
        list.append(Neuron(num_inputs))
    return list


def back_propagate(layers, output_layer, target_value):
    pass
    #for neuron in output_layer:
    #    neuron.Error = mf.getOutputError(neuron.output, target_value)
    #for layer in layers:
    #   for neuron in layer:
    #        neuron.



def main():
    inputs = pd.read_csv("C:\\Users\\bradl\\Documents\\Github\\Senior_Project-Neural_Net\\letter-recognition.txt")
    output_layer = len(set(inputs.letter))
    letters = inputs.letter
    newinputs = inputs.loc[:, inputs.columns != 'letter']
    print(newinputs)
    num_layers = input("Enter number of hidden layers: ")
    layers = []
    for i in range(0, num_layers):
        layer = input("Enter number of Neurons for hidden layer " + str(i+1) + ": ")
        layers.append(layer)
    neural_network = create_network(layers, output_layer, len(newinputs.columns))
    print(len(neural_network))


if __name__ == "__main__":
    main()
