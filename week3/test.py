#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

import numpy as np

data = np.loadtxt('data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

print(X[0, :])
print(np.log(X[0, :]))
print(y)
print(-5 - y)