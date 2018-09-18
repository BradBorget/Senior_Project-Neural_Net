import random
import pandas as pd


class Neuron:
    def __init__(self, numWeights):
        self.weights = []
        for i in range(numWeights + 1):
            self.weights.append(random.uniform(-1, 1))
        self.input = 0
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
    inputs = pd.read_csv("C:\\Users\\bradl\\Documents\\Github\\Senior_Project-Neural_Net\\letter-recognition.txt")
    print(inputs)
    print(len(set(inputs.letter)))

    #results = []
    #with open('example.csv') as File:
    #    reader = csv.DictReader(File)
    #    for row in reader:
    #        results.append(row)



if __name__ == "__main__":
    main()