#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Regularized logistic regression

		sigmoid function
		map feature (create new features - polynomial of higher order)
		non-linear decision boundary
		cost function
		gradient function
		plot decision boundary
		plot dataset
		training accuracy
		experiment with varying lambda (regularisation parameter)
		vectorised implementation <- important
"""

import numpy as np
from matplotlib import pyplot
import math
from scipy import optimize

class Exercise2:
	def __init__(self):
		self.X = None
		self.y = None

	def loadData(self):
		data = np.loadtxt('data2.txt', delimiter=',')
		self.X = data[:, :2]
		self.y = data[:, 2]

	def plotData(self, X, y):
		fig = pyplot.figure()

		pos = y == 1
		neg = y == 0

		pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
		pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms='8', mec='k', mew=1)

		pyplot.xlabel('Microchip test 1')
		pyplot.ylabel('Microchip test 2')
		pyplot.legend(['y = 1', 'y = 0'], loc='upper right')

	def mapFeature(self, X1, X2, degree=6):
	    if X1.ndim > 0:
	        out = [np.ones(X1.shape[0])]
	    else:
	        out = [np.ones(1)]

	    for i in range(1, degree + 1):
	        for j in range(i + 1):
	            out.append((X1 ** (i - j)) * (X2 ** j))

	    if X1.ndim > 0:
	        return np.stack(out, axis=1)
	    else:
	        return np.array(out)

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

	def costFunctionReg(self, theta, X, y, lambda_):
		m = y.size

		J = (-1 / m) * (np.dot(y.T, np.log(self.sigmoid(np.dot(X, theta)))) + np.dot(1 - y.T, np.log(1 - self.sigmoid(np.dot(X, theta)))))
		J += (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)

		grad = (1 / m) * np.dot(X.T, self.sigmoid(np.dot(X, theta)) - y)
		grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

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
	                z[i, j] = np.dot(self.mapFeature(ui, vj), theta)

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
		self.X = self.mapFeature(self.X[:, 0], self.X[:, 1])

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# initial_theta = np.zeros(self.X.shape[1])
		# lambda_ = 1
		# cost, grad = self.costFunctionReg(initial_theta, self.X, self.y, lambda_)

		# print('Cost at initial theta (zeros): {:.3f}'.format(cost))
		# print('Expected cost (approx)       : 0.693\n')

		# print('Gradient at initial theta (zeros) - first five values only:')
		# print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
		# print('Expected gradients (approx) - first five values only:')
		# print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

		# test_theta = np.ones(self.X.shape[1])
		# cost, grad = self.costFunctionReg(test_theta, self.X, self.y, 10)

		# print('------------------------------\n')
		# print('Cost at test theta    : {:.2f}'.format(cost))
		# print('Expected cost (approx): 3.16\n')

		# print('Gradient at test theta - first five values only:')
		# print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
		# print('Expected gradients (approx) - first five values only:')
		# print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		initial_theta = np.zeros(self.X.shape[1])
		options = {'maxiter': 1000}
		lambda_ = 1
		res = optimize.minimize(self.costFunctionReg, initial_theta, (self.X, self.y, lambda_), jac=True, method='TNC', options=options)
		cost = res.fun
		theta = res.x
		
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.plotDecisionBoundary(self.plotData, theta, self.X, self.y)
		pyplot.xlabel('Microchip Test 1')
		pyplot.ylabel('Microchip Test 2')
		pyplot.legend(['y = 1', 'y = 0'])
		pyplot.grid(False)
		pyplot.title('lambda = %0.2f' % lambda_)

		pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		p = self.predict(theta)

		print('Train Accuracy: %.1f %%' % (np.mean(p == self.y) * 100))
		print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')



if __name__ == '__main__':
	Exercise2().run()