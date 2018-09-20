import random
import pandas as pd


class Neuron:
    def __init__(self, num_weights):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.uniform(-1, 1))
        #self.inputs = inputs
        #self.Bj = 0 #I can't remember what B sub j represents, so I'm commenting it out for now.  Putting in Error instead to represent the value that the neuron came to
        self.output = 0
        self.Error = 0


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



def main():
    inputs = pd.read_csv("C:\\Users\\bradl\\Documents\\Github\\Senior_Project-Neural_Net\\letter-recognition.txt")
    print(inputs)
    output_layer = len(set(inputs.letter))
    num_layers = input("Enter number of hidden layers: ")
    layers = []
    for i in range(0, num_layers):
        layer = input("Enter number of Neurons for hidden layer " + str(i+1) + ": ")
        layers.append(layer)
    neural_network = create_network(layers, output_layer, (len(inputs.columns) - 1))
    print(len(neural_network))


if __name__ == "__main__":
    main()
