#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

""" 
	Multivariate linear regression
"""

import numpy as np
from matplotlib import pyplot


class Exercise2:
	def __init__(self):
		self.X = None
		self.y = None
		self.m = 0

	def loadData(self):
		data = np.loadtxt('data2.txt', delimiter=',')
		self.X = data[:, :2]
		self.y = data[:, 2]
		self.m = self.y.size

	def featureNormalize(self, X):
		X_norm = X.copy()

		mu = np.zeros(X.shape[1])
		sigma = np.zeros(X.shape[1])

		all_means = np.mean(X, 0)
		all_stds = np.std(X, 0)

		for i in range(X.shape[1]):
			mu[i] = all_means[i]
			sigma[i] = all_stds[i]

			for j in range(X.shape[0]):
				X_norm[j][i] = (X[j][i] - mu[i]) / sigma[i]

		return X_norm, mu, sigma

	def addColumnOfOnes(self, X):
		self.X = np.concatenate([np.ones((self.m, 1)), X], axis=1)

	def computeCostMulti(self, theta):
		J = np.dot((np.dot(self.X, theta) - self.y).T, np.dot(self.X, theta) - self.y) / (2 * self.m)
		return J

	def gradientDescentMulti(self, theta, alpha, num_iters):
		n = self.X.shape[1] - 1
		theta = theta.copy()
		J_history = []

		for i in range(num_iters):
			delta = 0
			for i in range(self.m):
				delta += np.dot((np.dot(theta, self.X[i]) - self.y[i]), self.X[i])
			delta /= self.m
			theta = theta - alpha * delta

			J_history.append(self.computeCostMulti(theta))

		return theta, J_history

	def normalEqn(self):
		theta = np.zeros(self.X.shape[1])

		theta = np.linalg.inv(np.dot(self.X.T, self.X))
		theta = np.dot(theta, self.X.T)
		theta = np.dot(theta, self.y)

		return theta

	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		X_norm, mu, sigma = self.featureNormalize(self.X)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.addColumnOfOnes(X_norm)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		alpha = 2.5
		num_alpha_steps = 6
		num_iters = 50
		pyplot.xlabel('Gradient descent iterations')
		pyplot.ylabel('Cost function')
		pyplot.title('Cost function of multivariate linear regression for different learning rates')
		labels = []

		for _ in range(num_alpha_steps):
			alpha /= 3
			theta, J_history = self.gradientDescentMulti(np.ones(self.X.shape[1]), alpha, num_iters)

			pyplot.plot(range(0, len(J_history)), J_history)
			labels.append('alpha = {:5f}'.format(alpha))

		pyplot.legend(labels)
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		alpha = 0.83
		num_iters = 1500
		theta, J_history = self.gradientDescentMulti(np.ones(self.X.shape[1]), alpha, num_iters)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		A = np.array([[1650.0, 3.0]])

		for i in range(A.shape[1]):
			A[0][i] = (A[0][i] - mu[i]) / sigma[i]

		A = np.concatenate([np.ones((1, 1)), A], axis=1)

		print("A 1650 sq feet house with 3 bedrooms should sell for ${:.0f}".format(np.dot(A, theta)[0]))

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()
		self.addColumnOfOnes(self.X)
		theta = self.normalEqn()

		A = np.array([[1650.0, 3.0]])
		A = np.concatenate([np.ones((1, 1)), A], axis=1)

		print("A 1650 sq feet house with 3 bedrooms should sell for ${:.0f}".format(np.dot(A, theta)[0]))


if __name__ == '__main__':
	Exercise2().run()