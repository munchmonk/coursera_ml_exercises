#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat


class Exercise2:
	def __init__(self):
		self.X = None
		self.y = None
		self.m = 0

		self.Theta1 = None
		self.Theta2 = None

	def loadData(self):
		data = loadmat("ex3data1.mat")
		self.X = data['X']
		self.y = data['y'].ravel()
		self.y[self.y == 10] = 0
		self.m = self.y.size
		
	def visualise_random_entries(self):
		rand_indices = np.random.choice(self.m, 100, replace=False)
		sel = self.X[rand_indices, :]
		self.displayData(sel)

	def load_precalculated_weights(self):
		input_layer_size = 400
		hidden_layer_size = 25
		num_labels = 10

		weights = loadmat("ex3weights.mat")
		self.Theta1 = weights["Theta1"]
		self.Theta2 = weights["Theta2"]

		self.Theta2 = np.roll(self.Theta2, 1, axis=0) # MATLAB glitch

	def predict(self, Theta1, Theta2, X):
		if X.ndim == 1:
			X = X[None]
		m = X.shape[0]
		num_labels = Theta2.shape[0]
		p = np.zeros(X.shape[0])

		X = np.concatenate([np.ones((m, 1)), X], axis=1)

		a_2 = self.sigmoid(np.dot(Theta1, X.T))
		a_2 = np.concatenate([np.ones((1, a_2.shape[1])), a_2], axis=0)

		a_3 = self.sigmoid(np.dot(Theta2, a_2))

		p = np.argmax(a_3, axis=0)

		return p


	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.visualise_random_entries()
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.load_precalculated_weights()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# pred = self.predict(self.Theta1, self.Theta2, self.X)

		# print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == self.y) * 100))
		# print('Expected: 97.5%')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		indices = np.random.permutation(self.m)
		max_examples = 20
		indices = indices[:max_examples]

		while indices.size > 0:
			i, indices = indices[0], indices[1:]
			self.displayData(self.X[i, :], figsize=(4, 4))
			pred = self.predict(self.Theta1, self.Theta2, self.X[i, :])
			print('Neural network prediction: {}'.format(*pred))
			pyplot.show()
		else:
			print('No more images to display!')

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

	def sigmoid(self, z):
	    """
	    Computes the sigmoid of z.
	    """
	    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
	Exercise2().run()