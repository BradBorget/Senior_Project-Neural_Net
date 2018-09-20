import random
import pandas as pd


class Neuron:
    def __init__(self, num_weights, inputs):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.uniform(-1, 1))
        self.input = inputs
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


def main():
    inputs = pd.read_csv("C:\\Users\\bradl\\Documents\\Github\\Senior_Project-Neural_Net\\letter-recognition.txt")
    print(inputs)
    print(len(set(inputs.letter)))

    #results = []
    #with open('example.csv') as File:
    #    reader = csv.DictReader(File)
    #    for row in reader:
    #        results.append(row)

    results = []
    with open('letter-recognition.txt') as File:
        reader = pd.read_csv(File)
        for row in reader:
            results.append(row)
    num_weights = len(set(reader.letter))
    num_layers = input("Enter number of hidden layers: ")
    output_layer = 8
    layers = []
    for i in range(0, num_layers):
        layer = input("Enter number of Neurons for hidden layer " + str(i+1) + ": ")
        layers.append(layer)
    neural_network = create_network(layers, output_layer, num_weights, results)
    print(len(neural_network))


def create_network(layers, hidden_layer, num_weight, inputs):
    neuron_list = []
    for i in layers:
        neuron = Neuron(num_weight, inputs)
        neuron_list.append(neuron)
    neuron_list.append(hidden_layer)
    return neuron_list


if __name__ == "__main__":
    main()
