#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Multiclass logistic regression
		one vs all
		loading matlab data
		viualising random examples
		regularised logistic regression
		cost function
		gradient
		inline sigmoid function
		predictions on training set - training accuracy
		built-int optimize function to calculate logistic regression parameters
		argmax returns the index of the largest element in an array, great for finding most likely prediction amongst all the predictions for all the classes
"""

import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat


class Exercise1:
	def __init__(self):
		self.X = None
		self.y = None
		self.m = 0

	def loadData(self):
		self.num_labels = 10
		data = loadmat("ex3data1.mat")
		self.X = data['X']
		self.y = data['y'].ravel()

		self.y[self.y == 10] = 0
		self.m = self.y.size

	def visualise_random_entries(self):
		rand_indices = np.random.choice(self.m, 100, replace=False)
		sel = self.X[rand_indices, :]
		self.displayData(sel)

	def test_values(self):
		theta_t = np.array([-2, -1, 1, 2], dtype=float)
		X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
		y_t = np.array([1, 0, 1, 0, 1])
		lambda_t = 3

		J, grad = self.lrCostFunction(theta_t, X_t, y_t, lambda_t)
		print('Cost         : {:.6f}'.format(J))
		print('Expected cost: 2.534819')
		print('-----------------------')
		print('Gradients:')
		print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
		print('Expected gradients:')
		print(' [0.146561, -0.548558, 0.724722, 1.398003]');

	def lrCostFunction(self, theta, X, y, lambda_):
		m = y.size

		if y.dtype == bool:
			y = y.astype(int)

		J = (-1 / m) * (np.dot(y.T, np.log(self.sigmoid(np.dot(X, theta)))) + np.dot(1 - y.T, np.log(1 - self.sigmoid(np.dot(X, theta)))))
		J += (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)

		grad = (1 / m) * np.dot(X.T, self.sigmoid(np.dot(X, theta)) - y)
		grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

		return J, grad

	def oneVsAll(self, X, y, num_labels, lambda_):
		m, n = X.shape
		all_theta = np.zeros((num_labels, n + 1))

		X = np.concatenate([np.ones((m, 1)), X], axis=1)

		for i in range(num_labels):
			initial_theta = np.zeros(n + 1)
			options = {"maxiter": 50}
			res = optimize.minimize(self.lrCostFunction, initial_theta, (X, (y == i), lambda_), jac=True, method='CG', options=options)
			theta = res.x
			all_theta[i] = theta

		return all_theta

	def predictOneVsAll(self, all_theta, X):
		m = X.shape[0]
		num_labels = all_theta.shape[0]

		p = np.zeros(m)

		X = np.concatenate([np.ones((m, 1)), X], axis=1)

		predictions_matrix = np.dot(X, all_theta.T)
		p = np.argmax(predictions_matrix, axis=1)

		return p

	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.visualise_random_entries()
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.test_values()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		lambda_ = 0.1
		all_theta = self.oneVsAll(self.X, self.y, self.num_labels, lambda_)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		pred = self.predictOneVsAll(all_theta, self.X)
		print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == self.y) * 100))
		print('Expected: 95.1%')

	def sigmoid(self, z):
	    """
	    Computes the sigmoid of z.
	    """
	    return 1.0 / (1.0 + np.exp(-z))

	def displayData(self, X, example_width=None, figsize=(10, 10)):
	    """
	    Displays 2D data stored in X in a nice grid.
	    """
	    # Compute rows, cols
	    if X.ndim == 2:
	        m, n = X.shape
	    elif X.ndim == 1:
	        n = X.size
	        m = 1
	        X = X[None]  # Promote to a 2 dimensional array
	    else:
	        raise IndexError('Input X should be 1 or 2 dimensional.')

	    example_width = example_width or int(np.round(np.sqrt(n)))
	    example_height = n / example_width

	    # Compute number of items to display
	    display_rows = int(np.floor(np.sqrt(m)))
	    display_cols = int(np.ceil(m / display_rows))

	    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
	    fig.subplots_adjust(wspace=0.025, hspace=0.025)

	    ax_array = [ax_array] if m == 1 else ax_array.ravel()

	    for i, ax in enumerate(ax_array):
	        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
	                  cmap='Greys', extent=[0, 1, 0, 1])
	        ax.axis('off')


if __name__ == "__main__":
	Exercise1().run()