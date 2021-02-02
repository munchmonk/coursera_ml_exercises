#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

import numpy as np

data = np.loadtxt('data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

print(np.sum((y[1:]+5) ** 2))
print(3160/y.size)
