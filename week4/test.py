#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3


import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.ones((1, A.shape[1]))

C = np.concatenate([b, A], axis=0)

print(C)