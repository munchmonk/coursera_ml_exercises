#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Linear regression with one variable
"""

import numpy as np
from matplotlib import pyplot


class Exercise1:
	def __init__(self):
		self.X = None
		self.y = None
		self.m = 0

	def loadData(self):
		data = np.loadtxt(('data1.txt'), delimiter=',')
		self.X = data[:, 0]
		self.y = data[:, 1]
		self.m = self.y.size		

	def plotData(self, X, y):
		fig = pyplot.figure()
		pyplot.plot(X, y, 'ro', ms=10, mec='k')
		pyplot.xlabel('City population in 10k')
		pyplot.ylabel('Profit per truck in $10k')

	def addColumnOfOnes(self):
		self.X = np.stack([np.ones(self.m), self.X], axis=1)

	def computeCost(self, theta):
		J = 0
		for i in range(self.m):
			h = np.dot(theta, self.X[i])
			J += (h - self.y[i]) ** 2
		J /= (2 * self.m)

		return J

	def gradientDescent(self, theta, alpha, num_iters):
		theta = theta.copy()
		J_history = []

		for i in range(num_iters):
			delta = 0
			for i in range(self.m):
				delta += np.dot((np.dot(theta, self.X[i]) - self.y[i]), self.X[i])
			delta /= self.m
			theta = theta - alpha * delta

			J_history.append(self.computeCost(theta))

		return theta, J_history

	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()
		
		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotData(self.X, self.y)
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.addColumnOfOnes()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		starting_theta = np.array([0, 0])
		alpha = 0.01
		num_iters = 1500
		theta, J_history = self.gradientDescent(starting_theta, alpha, num_iters)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotData(self.X[:, 1], self.y)
		# pyplot.plot(self.X[:, 1], np.dot(self.X, theta), '-')
		# pyplot.legend(['Training data', 'Linear regression'])
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# predict1 = np.dot([1, 3.5], theta)
		# print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))
		# predict2 = np.dot([1, 7], theta)
		# print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

		# --------------------------------------------------------------------------------------------------------------------------------------------
		theta0_vals = np.linspace(-10, 10, 100)
		theta1_vals = np.linspace(-1, 4, 100)
		J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

		for i, theta0 in enumerate(theta0_vals):
			for j, theta1 in enumerate(theta1_vals):
				J_vals[i, j] = self.computeCost([theta0, theta1])

		J_vals = J_vals.T

		fig = pyplot.figure(figsize=(12, 5))
		ax = fig.add_subplot(121, projection='3d')
		ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
		pyplot.xlabel('theta0')
		pyplot.ylabel('theta1')
		pyplot.title('Surface')

		ax = pyplot.subplot(122)
		pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
		pyplot.xlabel('theta0')
		pyplot.ylabel('theta1')
		pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
		pyplot.title('Contour, showing minimum')

		pyplot.show()


if __name__ == '__main__':
	Exercise1().run()
