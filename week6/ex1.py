#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Regularised logistic regresison and bias vs. variance

		bias - variance tradeoff
		learning curves
		training error
		cross validation error
		test error
		mapping to polynomial
		validation curve
"""

import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import random


class Exercise1:
	def __init__(self):
		# Training set
		self.X = None
		self.y = None

		# Cross validation set
		self.Xval = None
		self.yval = None

		# Test set
		self.Xtest = None
		self.ytest = None

		self.m = 0

	def loadData(self):
		data = loadmat("ex5data1.mat")
		self.X, self.y = data['X'], data['y'][:, 0]
		self.Xtest, self.ytest = data['Xtest'], data['ytest'][:, 0]
		self.Xval, self.yval = data['Xval'], data['yval'][:, 0]
		self.m = self.y.size

		# self.addColumnOfOnes()

	def visualise_training_data(self):
		pyplot.plot(self.X, self.y, 'ro', ms=10, mec='k', mew=1)
		pyplot.xlabel('Change in water level (x)')
		pyplot.ylabel('Water flowing out of the dam (y)')

		pyplot.show()

	def linearRegCostFunction(self, X, y, theta, _lambda=0.0):
		m = y.size

		J = np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y) / (2 * m) + np.sum(theta[1:] ** 2) * (_lambda / (2 * m))
		grad = np.dot(X.T, np.dot(X, theta) - y) / m
		grad[1:] = grad[1:] + (_lambda / m) * theta[1:]
		
		return J, grad

	def plotLinearRegression(self):
		X_aug = np.concatenate([np.ones((self.m, 1)), self.X], axis=1)
		theta = self.trainLinearReg(self.linearRegCostFunction, X_aug, self.y, lambda_ = 0)

		pyplot.plot(self.X, self.y, 'ro', ms=10, mec='k', mew=1.5)
		pyplot.xlabel('Change in water level (x)')
		pyplot.ylabel('Water flowing out of the dam (y)')
		pyplot.plot(self.X, np.dot(X_aug, theta), '--', lw=2);

		pyplot.show()

	def learningCurve(self, X, y, Xval, yval, lambda_=0):
		m = y.size

		error_train = np.zeros(m)
		error_val = np.zeros(m)

		X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
		Xval_aug = np.concatenate([np.ones((Xval.shape[0], 1)), Xval], axis=1)

		for i in range(1, m + 1):
			theta = self.trainLinearReg(self.linearRegCostFunction, X_aug[:i, :], y[:i], lambda_)

			# The error is equal to the cost function without the regularisation parameter
			error_train[i - 1] = self.linearRegCostFunction(X_aug[:i, :], y[:i], theta, 0)[0]
			error_val[i - 1] = self.linearRegCostFunction(Xval_aug, yval, theta, 0)[0]

		return error_train, error_val

	def plotLearningCurve(self):
		error_train, error_val = self.learningCurve(self.X, self.y, self.Xval, self.yval, lambda_=0)

		pyplot.plot(np.arange(1, self.m + 1), error_train, np.arange(1, self.m + 1), error_val, lw=2)
		pyplot.title('Learning curve for linear regression')
		pyplot.legend(['Train', 'Cross Validation'])
		pyplot.xlabel('Number of training examples')
		pyplot.ylabel('Error')
		pyplot.axis([0, 13, 0, 150])

		print('# Training Examples\tTrain Error\tCross Validation Error')
		for i in range(self.m):
		    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

		pyplot.show()

	def polyFeatures(self, X, p):
		X_poly = np.zeros((X.shape[0], p))

		for i in range(p):
			X_poly[:, i] = X[:, 0] ** (i + 1)

		return X_poly

	def mapToPolynomialAndNormalise(self, X, Xtest, ytest, Xval, yval, p):
		# Map X onto Polynomial Features and Normalize
		X_poly = self.polyFeatures(X, p)
		X_poly, mu, sigma = self.featureNormalize(X_poly)
		X_poly = np.concatenate([np.ones((self.m, 1)), X_poly], axis=1)

		# Map X_poly_test and normalize (using mu and sigma)
		X_poly_test = self.polyFeatures(Xtest, p)
		X_poly_test -= mu
		X_poly_test /= sigma
		X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

		# Map X_poly_val and normalize (using mu and sigma)
		X_poly_val = self.polyFeatures(Xval, p)
		X_poly_val -= mu
		X_poly_val /= sigma
		X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

		# print('Normalized Training Example 1:', X_poly[0, :])

		return X_poly, X_poly_val, X_poly_test, mu, sigma

	def plotPolynomialFit(self, X_poly, y, mu, sigma, p, X_poly_val, lambda_=0.0):
		theta = self.trainLinearReg(self.linearRegCostFunction, X_poly, y, lambda_=lambda_, maxiter=55)

		# Plot training data and fit
		pyplot.plot(self.X, y, 'ro', ms=10, mew=1.5, mec='k')

		self.plotFit(self.polyFeatures, np.min(self.X), np.max(self.X), mu, sigma, theta, p)

		pyplot.xlabel('Change in water level (x)')
		pyplot.ylabel('Water flowing out of the dam (y)')
		pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
		pyplot.ylim([-20, 50])

		pyplot.figure()
		error_train, error_val = self.learningCurve(X_poly, self.y, X_poly_val, self.yval, lambda_)
		pyplot.plot(np.arange(1, 1 + self.m), error_train, np.arange(1, 1 + self.m), error_val)

		pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
		pyplot.xlabel('Number of training examples')
		pyplot.ylabel('Error')
		pyplot.axis([0, 13, 0, 100])
		pyplot.legend(['Train', 'Cross Validation'])

		print('Polynomial Regression (lambda = %f)\n' % lambda_)
		print('# Training Examples\tTrain Error\tCross Validation Error')
		for i in range(self.m):
		    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

	def validationCurve(self, X, y, Xval, yval):
		lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

		error_train = np.zeros(len(lambda_vec))
		error_val = np.zeros(len(lambda_vec))

		for i in range(len(lambda_vec)):
			theta = self.trainLinearReg(self.linearRegCostFunction, X, y, lambda_=lambda_vec[i], maxiter=55)
			error_train[i] = self.linearRegCostFunction(X, y, theta, 0)[0]
			error_val[i] = self.linearRegCostFunction(Xval, yval, theta, 0)[0]

		return lambda_vec, error_train, error_val

	def computeTestError(self, X, y, X_poly_test, lambda_):
		theta = self.trainLinearReg(self.linearRegCostFunction, X, y, lambda_=lambda_, maxiter=55)
		error_test = self.linearRegCostFunction(X_poly_test, self.ytest, theta, 0)[0]
		return error_test

	def learningCurvesRandomExamples(self, X, y, Xval, yval, lambda_, iters):
		m = y.size

		avg_error_train = np.zeros(m)
		avg_error_val = np.zeros(m)

		for i in range(1, m + 1):
			error_train = 0
			error_val = 0

			for _ in range(iters):
				X_random_range = random.sample(range(X.shape[0]), i)
				Xval_random_range = random.sample(range(Xval.shape[0]), i)

				X_shuffled = X[X_random_range]
				y_shuffled = y[X_random_range]
				Xval_shuffled = Xval[Xval_random_range]
				yval_shuffled = yval[Xval_random_range]

				theta = self.trainLinearReg(self.linearRegCostFunction, X[X_random_range, :], y[X_random_range], lambda_)

				error_train += self.linearRegCostFunction(X[X_random_range, :], y[X_random_range], theta, 0)[0]
				error_val += self.linearRegCostFunction(Xval[Xval_random_range, :], yval[Xval_random_range], theta, 0)[0]

			avg_error_train[i - 1] = error_train / iters
			avg_error_val[i - 1] = error_val / iters

		pyplot.plot(np.arange(1, m + 1), avg_error_train, np.arange(1, m + 1), avg_error_val, lw=2)
		pyplot.title('Learning curve for linear regression')
		pyplot.legend(['Train', 'Cross Validation'])
		pyplot.xlabel('Number of training examples')
		pyplot.ylabel('Error')
		pyplot.axis([0, 13, 0, 150])

		print('# Training Examples\tTrain Error\tCross Validation Error')
		for i in range(m):
		    print('  \t%d\t\t%f\t%f' % (i+1, avg_error_train[i], avg_error_val[i]))

		pyplot.show()

	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.visualise_training_data()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test cost function

		# theta = np.array([1, 1])
		# J, grad = self.linearRegCostFunction(np.concatenate([np.ones((self.m, 1)), self.X], axis=1), self.y, theta, 1)

		# print('Cost at theta = [1, 1]:\t   %f ' % J)
		# print('This value should be about 303.993192)\n' % J)

		# print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
		# print(' (this value should be about [-15.303016, 598.250744])\n')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotLinearRegression()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotLearningCurve()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Add polynomial features

		p = 8
		X_poly, X_poly_val, X_poly_test, mu, sigma = self.mapToPolynomialAndNormalise(self.X, self.Xtest, self.ytest, self.Xval, self.yval, p)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.plotPolynomialFit(X_poly, self.y, mu, sigma, p, X_poly_val, 0)
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Try different values of lambda

		# lambdas = [0.01, 1, 20, 60, 100]
		# for lambda_ in lambdas:
		# 	self.plotPolynomialFit(X_poly, self.y, mu, sigma, p, X_poly_val, lambda_)
		# 	pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Try different values of lambda and plot the training error and cross validation error to choose a suitable one

		# lambda_vec, error_train, error_val = self.validationCurve(X_poly, self.y, X_poly_val, self.yval)

		# pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
		# pyplot.legend(['Train', 'Cross Validation'])
		# pyplot.xlabel('lambda')
		# pyplot.ylabel('Error')

		# print('lambda\t\tTrain Error\tValidation Error')
		# for i in range(len(lambda_vec)):
		#     print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Compute test error for lambda = 3, chosen by looking at the previous graph

		# lambda_ = 3
		# error_test = self.computeTestError(X_poly, self.y, X_poly_test, lambda_)
		# print('Test error:     {}'.format(error_test))
		# print('Expected error: 3.8599')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Randomly select training examples and cross validation set examples 50 times and compute the average cost
		lambda_ = 0.01
		iters = 50
		self.learningCurvesRandomExamples(X_poly, self.y, X_poly_val, self.yval, lambda_, iters)

	def trainLinearReg(self, linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
	    """
	    Trains linear regression using scipy's optimize.minimize.

	    Parameters
	    ----------
	    X : array_like
	        The dataset with shape (m x n+1). The bias term is assumed to be concatenated.

	    y : array_like
	        Function values at each datapoint. A vector of shape (m,).

	    lambda_ : float, optional
	        The regularization parameter.

	    maxiter : int, optional
	        Maximum number of iteration for the optimization algorithm.

	    Returns
	    -------
	    theta : array_like
	        The parameters for linear regression. This is a vector of shape (n+1,).
	    """
	    # Initialize Theta
	    initial_theta = np.zeros(X.shape[1])

	    # Create "short hand" for the cost function to be minimized
	    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

	    # Now, costFunction is a function that takes in only one argument
	    options = {'maxiter': maxiter}

	    # Minimize using scipy
	    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
	    return res.x

	def featureNormalize(self, X):
	    """
	    Normalizes the features in X returns a normalized version of X where the mean value of each
	    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
	    working with learning algorithms.

	    Parameters
	    ----------
	    X : array_like
	        An dataset which is a (m x n) matrix, where m is the number of examples,
	        and n is the number of dimensions for each example.

	    Returns
	    -------
	    X_norm : array_like
	        The normalized input dataset.

	    mu : array_like
	        A vector of size n corresponding to the mean for each dimension across all examples.

	    sigma : array_like
	        A vector of size n corresponding to the standard deviations for each dimension across
	        all examples.
	    """
	    mu = np.mean(X, axis=0)
	    X_norm = X - mu

	    sigma = np.std(X_norm, axis=0, ddof=1)
	    X_norm /= sigma
	    return X_norm, mu, sigma

	def plotFit(self, polyFeatures, min_x, max_x, mu, sigma, theta, p):
	    """
	    Plots a learned polynomial regression fit over an existing figure.
	    Also works with linear regression.
	    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).

	    Parameters
	    ----------
	    polyFeatures : func
	        A function which generators polynomial features from a single feature.

	    min_x : float
	        The minimum value for the feature.

	    max_x : float
	        The maximum value for the feature.

	    mu : float
	        The mean feature value over the training dataset.

	    sigma : float
	        The feature standard deviation of the training dataset.

	    theta : array_like
	        The parameters for the trained polynomial linear regression.

	    p : int
	        The polynomial order.
	    """
	    # We plot a range slightly bigger than the min and max values to get
	    # an idea of how the fit will vary outside the range of the data points
	    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

	    # Map the X values
	    X_poly = polyFeatures(x, p)
	    X_poly -= mu
	    X_poly /= sigma

	    # Add ones
	    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

	    # Plot
	    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)


if __name__ == '__main__':
	Exercise1().run()