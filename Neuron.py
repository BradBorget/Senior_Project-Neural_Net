import random
import pandas


class Neuron:
    def __init__(self, num_weights, inputs):
        self.weights = []
        for i in range(num_weights + 1):
            self.weights.append(random.uniform(-1, 1))
        self.input = inputs
        self.Bj = 0


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
    results = []
    with open('letter-recognition.txt') as File:
        reader = pandas.read_csv(File)
        for row in reader:
            results.append(row)
    num_weights = len(set(reader.letter))
    num_layers = input("Enter number of layers: ")
    hidden_layer = 8
    layers = []
    for i in range(0, num_layers):
        layer = input("Enter number of Neurons for layer " + str(i+1) + ": ")
        layers.append(layer)
    neural_network = create_network(layers, hidden_layer, num_weights, results)
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
