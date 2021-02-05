#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3


import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


a = 0
b = np.array([0, 1])
c = np.array([[0, 1], [-1, 2]])

print(sigmoid(a))
print(sigmoid(b))
print(sigmoid(c))