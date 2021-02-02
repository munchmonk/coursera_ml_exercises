#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Optional exercise
	Multivariate linear regression 
"""

import numpy as np
from matplotlib import pyplot

data = np.loadtxt('data2.txt', delimiter=',')

# X: 47x2 matrix, column 0 is the size in sq feet of a house, column 1 the number of bedroos
X = data[:, :2]

# y: 47x1 vector, containing the associated price for which the house was sold
y = data[:, 2]

# m: number of training examples
m = y.size

# Print some data points
# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:,1]', 'y'))
# print('-'*26)
# for i in range(10):
# 	print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

def featureNormalize(X):
	"""
		Normalizes a matrix X so that each feature has mean 0 and standard deviation 1. Ref. feature scaling.
		The mean of a feature is subtracted from each entry and then divided by its standard deviation.
		N.B. this must be done individually for each feature (i.e. each column of X)
		Mean and standard deviation must both be saved so that this can be applied to future entries to make a prediction based on this model. 
	"""

	X_norm = X.copy()

	# Mean
	mu = np.zeros(X.shape[1])

	# Standard deviation
	sigma = np.zeros(X.shape[1])

	all_means = np.mean(X, 0)
	all_stds = np.std(X, 0)

	# Loop over features
	for i in range(X.shape[1]):
		mu[i] = all_means[i]
		sigma[i] = all_stds[i]

		# Loop over each single data entry in a given feature
		for j in range(X.shape[0]):
			X_norm[j][i] = (X[j][i] - mu[i]) / sigma[i]

	return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)

# Now that X has been normalized we add the intercept term (a column of ones):
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


def computeCostMulti(X, y, theta):
	"""
	Compute cost for linear regression with multiple variables for a certain theta.

	m: number of training examples
	n: number of features (n > 1 in this case since it's multivariate linear regression we are doing)
	X: m x n+1 matrix containing training examples
	y: vector of length m containing the values for a given data point
	theta: vector of length n+1 containing the current parameters of our linear regression hypothesis

	Returns J, a scalar, the value of the cost function for this particular choice of theta

	Formula: J(theta) = 1/(2*m) * (X*theta - y).T * (X*theta - y)
	"""

	m = y.shape[0]

	J = 0

	tmp = np.dot(X, theta) - y
	J = np.dot(tmp.T, tmp)
	J /= 2 * m

	return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
	"""
	Performs gradient descent to find a suitable theta.

	num_iters: number of iterations
	alpha: learning rate
	X, y, theta: as per computeCostMulti

	Returns theta (vector of length n+1) and J_history (a list)

	Formula: in each iteration, update theta as following:
		theta = theta - (alpha / m) * (sum(h(x) - y) * x) [sum for i : 1 -> m]
		where h(x) = theta.T * X
	"""

	m = y.shape[0]
	n = X.shape[1] - 1

	# Arrays are passed by reference; we don't want to overwrite the original
	theta = theta.copy()

	J_history = []

	for i in range(num_iters):
		res = np.zeros(n + 1)
		for i in range(m):
			h = np.dot(theta, X[i])
			res += np.dot(h - y[i], X[i])
		res *= alpha / m
		theta = theta - res

		J_history.append(computeCostMulti(X, y, theta))

	return theta, J_history

# theta, J_history = gradientDescentMulti(X, y, [-1, -2, -3], 0.005, 1500)

# # -----------------------------------
# # Plot the history of the cost function J for different values of the learning rate alpha (empirically, J diverges for alpha >= 1.6)

# alpha = 2.5
# pyplot.xlabel('Gradient descent iterations')
# pyplot.ylabel('Cost function')
# pyplot.title('Cost function of multivariate linear regression for different learning rates')
# labels = []

# for _ in range(6):
# 	alpha /= 3
# 	theta, J_history = gradientDescentMulti(X, y, [1, 1, 1], alpha, 50)

# 	pyplot.plot(range(0, len(J_history)), J_history)
# 	labels.append('alpha = {:5f}'.format(alpha))

# pyplot.legend(labels)
# pyplot.show()

# # -----------------------------------
# # Use a good learning rate we empirically found -> alpha = 0.85 to calculate our parameter theta
# theta, J_history = gradientDescentMulti(X, y, [1, 1, 1], 0.85, 1500)

# # -----------------------------------
# # Use our hypothesis to calculate the value of a specific example: a house of 1650 sq feet and 3 bedrooms (remember to normalize the features!)
# A = np.array([[1650.0, 3.0]])

# # Normalize A
# for i in range(A.shape[1]):
# 	A[0][i] = (A[0][i] - mu[i]) / sigma[i]

# # Add the intercept (i.e. first term is 1)
# A = np.concatenate([np.ones((1, 1)), A], axis=1)

# # Calculate the estimate
# print("A 1650 sq feet house with 3 bedrooms should sell for ${:.2f}".format(np.dot(A, theta)[0]))

# # -----------------------------------
# Normal equation: solve multivariate linear regression in one calculation without need for looping, feature scaling, or choosing a learning rate
# "closed-form solution"

# Reload the data in case it's been modified
data = np.loadtxt('data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# We don't need to do feature scaling but we still need to add a column of 1s to the beginning of X to find the intercept term
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
	"""
	Returns theta, the parameter of our linear hypothesis
	Formula: theta = (X*X.T)^-1*X.T*y
	"""

	theta = np.zeros(X.shape[1])

	theta = np.linalg.inv(np.dot(X.T, X))
	theta = np.dot(theta, X.T)
	theta = np.dot(theta, y)

	return theta

theta = normalEqn(X, y)

# Recreate the matrix from the previous example: a 1650 sq ft house with 3 bedrooms
A = np.array([[1650.0, 3.0]])
A = np.concatenate([np.ones((1, 1)), A], axis=1)

# Calculate the expected selling price. This result matches the one found with gradient descent, meaning they both work
print("A 1650 sq feet house with 3 bedrooms should sell for ${:.2f}".format(np.dot(A, theta)[0]))















































