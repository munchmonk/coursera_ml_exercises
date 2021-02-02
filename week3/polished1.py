#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Logistic regression

		sigmoid function applied to scalar, vectors and matrices
		cost function
		gradient function
		built-in optimise function
		plot data
		plot decision boundary
		training accuracy
"""

import numpy as np
from matplotlib import pyplot
from scipy import optimize
import math


class Exercise2:
	def __init__(self):
		self.X = None
		self.y = None
		self.m = 0
		self.n = 0

	def loadData(self):
		data = np.loadtxt('data1.txt', delimiter=',')
		self.X = data[:, :2]
		self.y = data[:, 2]
		self.m, self.n = self.X.shape

	def plotData(self, X, y):
		fig = pyplot.figure()

		pos = y == 1
		neg = y == 0

		pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
		pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms='8', mec='k', mew=1)

		pyplot.xlabel('Results of first exam')
		pyplot.ylabel('Results of second exam')
		pyplot.legend(['Admitted', 'Not admitted'])

	def computeSigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def sigmoid(self, z):
		z = np.array(z)
		g = np.zeros(z.shape)
		dim = np.ndim(z)

		if dim == 0:
			g = np.array(self.computeSigmoid(z))

		elif dim == 1:
			for i in range(len(z)):
				g[i] = self.computeSigmoid(z[i])

		elif dim == 2:
			for i in range(z.shape[0]):
				for j in range(z.shape[1]):
					g[i][j] = self.computeSigmoid(z[i][j])

		return g

	def addColumnOfOnes(self):
		self.X = np.concatenate([np.ones((self.m, 1)), self.X], axis=1)

	def costFunction(self, theta, X, y):
		m = y.size

		J = (-1 / m) * (np.dot(y.T, np.log(self.sigmoid(np.dot(X, theta)))) + np.dot(1 - y.T, np.log(1 - self.sigmoid(np.dot(X, theta)))))

		grad = (1 / m) * np.dot(X.T, self.sigmoid(np.dot(X, theta)) - y)

		return J, grad

	def plotDecisionBoundary(self, plotData, theta, X, y):
	    theta = np.array(theta)

	    plotData(X[:, 1:3], y)

	    if X.shape[1] <= 3:
	        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
	        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
	        pyplot.plot(plot_x, plot_y)

	        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
	        pyplot.xlim([30, 100])
	        pyplot.ylim([30, 100])
	    else:
	        u = np.linspace(-1, 1.5, 50)
	        v = np.linspace(-1, 1.5, 50)

	        z = np.zeros((u.size, v.size))
	        for i, ui in enumerate(u):
	            for j, vj in enumerate(v):
	                z[i, j] = np.dot(mapFeature(ui, vj), theta)

	        z = z.T  

	        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
	        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

	def predict(self, theta):
		p = self.sigmoid(np.dot(self.X, theta)) > 0.5
		return p

	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotData(self.X, self.y)
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.addColumnOfOnes()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# theta = np.zeros(self.n+1)
		# print('Cost: ', self.costFunction(theta, self.X, self.y)[0])
		# print('Expected cost (approx): 0.693\n')
		# print('Grad:', self.costFunction(theta, self.X, self.y)[1])
		# print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

		# theta = np.array([-24, 0.2, 0.2])
		# print('Cost: ', self.costFunction(theta, self.X, self.y)[0])
		# print('Expected cost (approx): 0.218\n')
		# print('Grad:', self.costFunction(theta, self.X, self.y)[1])
		# print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		initial_theta = np.zeros(self.n + 1)
		options = {'maxiter': 400}
		res = optimize.minimize(self.costFunction, initial_theta, (self.X, self.y), jac=True, method='TNC', options=options)

		cost = res.fun
		theta = res.x

		# print('Cost: ', cost)
		# print('Expected cost (approx): 0.203\n');
		# print('Theta; ', theta)
		# print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotDecisionBoundary(self.plotData, theta, self.X, self.y)
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		prob = self.sigmoid(np.dot([1, 45, 85], theta))
		print('For a student with scores 45 and 85 we predict an admission probability of {:.3f}'.format(prob))
		print('Expected value: 0.775 +/- 0.002\n')

		p = self.predict(theta)
		training_accuracy = np.mean(p == self.y) * 100
		print('Training accuracy: {:.2f}%'.format(training_accuracy))
		print('Expected accuracy (approx): 89.00 %')


if __name__ == '__main__':
	Exercise2().run()