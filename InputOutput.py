import pandas as pd

inputs = pd.read_csv("letter-recognition.txt")

print(len(set(inputs.letter)))
