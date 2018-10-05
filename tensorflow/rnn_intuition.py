"""Simple python code to understand the intuition behind RNN
References -
TensorFlow 1.x Deep Learning Recipes for Artificial Intelligence Applications - Chapter 2
Uses-
python 3.6.3
numpy 1.14.5
pandas 0.20.3
"""
import numpy as np
import pandas as pd

# Generating a sequence with 2 features and 10 indices (timesteps)
timesteps = 10
input_features = 2
out_features = 3
sequence = pd.DataFrame(data=np.random.randint(low=0, high=10, size=(timesteps, input_features)),
                        index=range(timesteps),
                        columns=["x" + str(i + 1) for i in range(input_features)])
sequence.index.name = "t"
print(sequence.head())

# Building the model
# initialize the variables
Wx = np.random.normal(size=(input_features, out_features))
Wh = np.random.normal(size=(out_features, out_features))
b = np.random.random(out_features)

def little_neural_net(inputs, state):
    return np.round(np.tanh(np.dot(inputs, Wx) + np.dot(state, Wh) + b), 4)

outputs = pd.Series(index=sequence.index, dtype="object")
state = np.zeros((out_features, ))

# Run the rnn model
for t in sequence.index:
    outputs.iloc[t] = little_neural_net(sequence.iloc[t], state)
    state = outputs.iloc[t]

print(outputs.head())