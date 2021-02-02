#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Linear regression with one variable - commented
"""

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt(('data1.txt'), delimiter=',')

# X = population of a city in 10k (now a vector but will be a matrix), y = profit per truck in $10k in that city
X, y = data[:, 0], data[:, 1]

# m = number of training examples
m = y.size

def plotData(X, y):
	fig = pyplot.figure()

	# 'ro' -> red circles, ms -> marker size, mec -> marker edge colour
	pyplot.plot(X, y, 'ro', ms=10, mec='k')
	pyplot.xlabel('City population in 10k')
	pyplot.ylabel('Truck profit in $10k')
	# pyplot.show()

# plotData(X, y)

# Creates a column vector of size m containing all 1; stacks it (joins it) with array x along axis 1 (columns; axis=0 -> rows)
# i.e. creates a new column
X = np.stack([np.ones(m), X], axis=1)

def computeCost(X, y, theta):
	"""
	X is the matrix containing our training examples, of size m x n+1 
		m = number of training examples
		n = number of features
	y is the vector of the value of the function at each data point, of size m
	theta is a vector of size n+1 containing the parameters for the regression function

	This function returns J, a scalar, the value of the cost function with parameter theta
	we are trying to minimise J
	"""

	m = y.size
	J = 0

	for i in range(0, m):
		h = np.dot(theta, X[i])
		J += (h - y[i]) ** 2

	J /= (2 * m)

	return J

# print(computeCost(X, y, np.array([-1, 2])))


def gradientDescent(X, y, theta, alpha, num_iters):
	"""
	X: m x n+1 matrix, input dataset
	y: vector of length m, value at given features
	theta: vector of length n+1, the parameters chosen fo our hypothesis
	alpha: scalar, the learning rate
	num_iters: int, number of iterations to run gradient descent for

	returns:
	theta: vector of shape n+1, updated linear regression parameter
	J_history: list of the values of the cost function after each iteration (which should be always decreasing)
	"""

	# m = number of training examples
	m = y.shape[0] 

	# copy theta since numpy arrays are passed by reference; avoid changing the original
	theta = theta.copy()

	J_history = []

	for i in range(num_iters):
		res = np.zeros(2)
		for i in range(m):
			h = np.dot(theta, X[i])
			res += np.dot(h - y[i], X[i])
		res *= alpha / m
		theta = theta - res

		J_history.append(computeCost(X, y, theta))

	return theta, J_history


theta, J_history = gradientDescent(X, y, np.array([0, 0]), 0.01, 1500)

# # -----------------------------------
# # Plot the training data and the linear regression hypothesis
# plotData(X[:, 1], y)
# pyplot.plot(X[:, 1], np.dot(X, theta), '-')
# pyplot.legend(['Training data', 'Linear regression'])
# pyplot.show()

# # -----------------------------------
# # Plot the history of J
# pyplot.plot(range(0, len(J_history)), J_history)
# pyplot.xlabel('Iterations')
# pyplot.ylabel('Cost')
# pyplot.legend('J history')
# pyplot.show()

# # -----------------------------------
# # Use the values we calculated to do some predictions
# predict1 = np.dot([1, 3.5], theta)
# print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))
# predict2 = np.dot([1, 7], theta)
# print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

# -----------------------------------
# Plot a 2D surface graph and contour graph of the cost function J for various values of theta0 and theta1
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Computer J_vals
for i, theta0 in enumerate(theta0_vals):
	for j, theta1 in enumerate(theta1_vals):
		J_vals[i, j] = computeCost(X, y, [theta0, theta1])

# We need to transpose J_vals because of the way the surf command is implemented, otherwise the axes will be flipped
J_vals = J_vals.T

# Surface plot
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')

# Contour plout
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')

pyplot.show()
























