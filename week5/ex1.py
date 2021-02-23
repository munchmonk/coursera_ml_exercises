#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Neural network
		3 layers (1 hidden, with 25 units), 400 input units, 10 classes
		forward propagation
		backpropagation
		regularisation
		map output labels to vector
		cost function 
		gradient
		sigmoid gradient
		randomise initial parameters
		visualise hidden layer
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot
from scipy import optimize

class Exercise1:
	def __init__(self):
		self.X = None
		self.y = None


	def loadData(self):
		self.input_layer_size = 400
		self.hidden_layer_size = 25
		self.num_labels = 10

		data = loadmat("ex4data1.mat")
		self.X = data['X']
		self.y = data['y'].ravel()

		self.y[self.y == 10] = 0 # in matlab arrays start at 1; we remap 10 to 0

		self.m = self.y.size

		self.Theta1 = None
		self.Theta2 = None
		self.nnparams = None

	def visualise_random_entries(self):
		rand_indices = np.random.choice(self.m, 100, replace=False)
		sel = self.X[rand_indices, :]
		self.displayData(sel)

	def load_precalculated_weights(self):
		weights = loadmat("ex4weights.mat")
		self.Theta1 = weights["Theta1"]
		self.Theta2 = weights["Theta2"]

		self.Theta2 = np.roll(self.Theta2, 1, axis=0) # MATLAB glitch

		self.nnparams = np.concatenate([self.Theta1.ravel(), self.Theta2.ravel()])

	def mapLabelsToVector(self, y, num_labels):
		labels_matrix = np.zeros((y.size, num_labels))

		for i in range(y.size):
			labels_matrix[i][y[i]] = 1

		return labels_matrix

	def nnCostFunction(self, nnparams, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.0):
		Theta1 = np.reshape(nnparams[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
		Theta2 = np.reshape(nnparams[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)))

		m = y.size

		J = 0
		Theta1_grad = np.zeros(Theta1.shape)
		Theta2_grad = np.zeros(Theta2.shape)


		# My code begins here
		y = self.mapLabelsToVector(y, num_labels)
		
		# vectorised - calculate hypothesis applied to all examples at once. Forward propagation
		a_1 = X
		a_1 = np.concatenate([np.ones((m, 1)), a_1], axis=1)

		z_2 = np.dot(a_1, Theta1.T)
		a_2 = self.sigmoid(z_2)
		a_2 = np.concatenate([np.ones((a_2.shape[0], 1)), a_2], axis=1)
	
		z_3 = np.dot(a_2, Theta2.T)
		a_3 = self.sigmoid(z_3)

		# Partially vectorised implementation of the cost function
		for i in range(m):
			J += np.dot(-y[i], np.log(a_3[i, :])) - np.dot((np.ones(num_labels) - y[i]), np.log(np.ones(num_labels) - a_3[i, :]))
		J /= m

		# Add regularisation term
		Theta1_nobias = Theta1[:, 1:]
		Theta2_nobias = Theta2[:, 1:]
		J += (lambda_ / (2 * m)) * (np.sum(np.square(Theta1_nobias)) + np.sum(np.square(Theta2_nobias)))

		# Calculate gradients using backpropagation - vectorised, following tutorial
		delta_3 = a_3 - y
		delta_2 = np.dot(delta_3, Theta2_nobias) * self.sigmoidGradient(z_2)
		Delta_1 = np.dot(delta_2.T, a_1)
		Delta_2 = np.dot(delta_3.T, a_2)
		
		Theta1_grad = Delta_1 / m
		Theta2_grad = Delta_2 / m

		# Add regularisation term to gradient
		Theta1_grad[:, 1:] += (lambda_ / m) * Theta1_nobias
		Theta2_grad[:, 1:] += (lambda_ / m) * Theta2_nobias

		grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

		# one-liner fully vectorised equivalent implementation of the cost function - calculate cost of all examples for all classes at once
		# we only need the sum of the elements along the diagonal (np.trace) as this is what we would calculate with a for loop
		# do it on paper with dimensions analysis to see what's happening, m=5000 and k=10; y=5000x10, a_3=10x5000
		# This is SLOWER because we are multiplying massive matrices and discarding most of the output as we are only interested in the diagonal.
		# But I think it's super cool it can be done in one line!

		# Pro code below:
		# J = np.trace(np.dot(-y, np.log(a_3)) - np.dot(np.ones(y.shape) - y, np.log(np.ones(a_3.shape) - a_3))) / m

		return J, grad

	def sigmoid(self, z):
	    return 1.0 / (1.0 + np.exp(-z))

	def sigmoidGradient(self, z):
		g = np.zeros(z.shape)
		sig_z = self.sigmoid(z)
		g = sig_z * (np.ones(z.shape) - sig_z)
		return g


	def randInitializeWeights(self, L_in, L_out, epsilon_init=0.12):
		# good choice for epsilon: sqrt(6)/sqrt(l_in+l_out)

		# theta(l) has shape S(l+1)x(Sl + 1) aka L_outx(L_in + 1)
		W = np.zeros((L_out, 1 + L_in))

		# return values in [-epsilon, epsilon]
		W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

		return W


	def run(self):
		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.loadData()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# self.visualise_random_entries()
		# pyplot.show()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		self.load_precalculated_weights()

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test cost function

		# J, _ = self.nnCostFunction(self.nnparams, self.input_layer_size, self.hidden_layer_size, self.num_labels, self.X, self.y)
		# print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
		# print('The cost should be about                   : 0.287629.')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test cost function regularised

		# lambda_ = 1
		# J, _ = self.nnCostFunction(self.nnparams, self.input_layer_size, self.hidden_layer_size, self.num_labels, self.X, self.y, lambda_)
		# print('Cost at parameters (loaded from ex4weights): %.6f' % J)
		# print('This value should be about                 : 0.383770.')

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test sigmoid gradient function

		# z = np.array([-1, -0.5, 0, 0.5, 1])
		# g = self.sigmoidGradient(z)
		# print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
		# print(g)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Randomly initialise parameters

		initial_Theta1 = self.randInitializeWeights(self.input_layer_size, self.hidden_layer_size)
		initial_Theta2 = self.randInitializeWeights(self.hidden_layer_size, self.num_labels)

		# unroll parameters into a vector
		initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test gradient calculation - gradient checking

		# self.checkNNGradients(self.nnCostFunction)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Test gradient calculation with regularisation - gradient checking

		# lambda_ = 3
		# self.checkNNGradients(self.nnCostFunction, lambda_)

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Now that we know our code works, train the network to find suitable parameters Theta1 and Theta2

		options = {"maxiter": 100}
		lambda_ = 1
		costFunction = lambda p: self.nnCostFunction(p, self.input_layer_size, self.hidden_layer_size, self.num_labels, self.X, self.y, lambda_)
		res = optimize.minimize(costFunction, initial_nn_params, jac=True, method='TNC', options=options)
		nn_params = res.x
		Theta1 = np.reshape(nn_params[:self.hidden_layer_size * (self.input_layer_size + 1)], (self.hidden_layer_size, (self.input_layer_size + 1)))
		Theta2 = np.reshape(nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):], (self.num_labels, (self.hidden_layer_size + 1)))

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Calculate the training accuracy of the parameters we just learned

		pred = self.predict(Theta1, Theta2, self.X)
		print("Training set accuracy: %f" % (np.mean(pred == self.y) * 100))

		# --------------------------------------------------------------------------------------------------------------------------------------------
		# Visualise the activation of the hidden layer - the i-th row of Theta1 contains all the weights for the i-th hidden unit. We display it as a
		# 20x20 image (we have 400 input features) (discard the bias unit) this shows us what activates each hidden unit

		# self.displayData(Theta1[:, 1:])
		# pyplot.show()




	def predict(self, Theta1, Theta2, X):
	    """
	    Predict the label of an input given a trained neural network
	    Outputs the predicted label of X given the trained weights of a neural
	    network(Theta1, Theta2)
	    """
	    # Useful values
	    m = X.shape[0]
	    num_labels = Theta2.shape[0]

	    # You need to return the following variables correctly
	    p = np.zeros(m)
	    h1 = self.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
	    h2 = self.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
	    p = np.argmax(h2, axis=1)
	    return p


	def debugInitializeWeights(self, fan_out, fan_in):
	    """
	    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
	    connections using a fixed strategy. This will help you later in debugging.

	    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
	    the "bias" terms.

	    Parameters
	    ----------
	    fan_out : int
	        The number of outgoing connections.

	    fan_in : int
	        The number of incoming connections.

	    Returns
	    -------
	    W : array_like (1+fan_in, fan_out)
	        The initialized weights array given the dimensions.
	    """
	    # Initialize W using "sin". This ensures that W is always of the same values and will be
	    # useful for debugging
	    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
	    W = W.reshape(fan_out, 1+fan_in, order='F')
	    return W
		

	def checkNNGradients(self, nnCostFunction, lambda_=0):
	    """
	    Creates a small neural network to check the backpropagation gradients. It will output the
	    analytical gradients produced by your backprop code and the numerical gradients
	    (computed using computeNumericalGradient). These two gradient computations should result in
	    very similar values.

	    Parameters
	    ----------
	    nnCostFunction : func
	        A reference to the cost function implemented by the student.

	    lambda_ : float (optional)
	        The regularization parameter value.
	    """
	    input_layer_size = 3
	    hidden_layer_size = 5
	    num_labels = 3
	    m = 5

	    # We generate some 'random' test data
	    Theta1 = self.debugInitializeWeights(hidden_layer_size, input_layer_size)
	    Theta2 = self.debugInitializeWeights(num_labels, hidden_layer_size)

	    # Reusing debugInitializeWeights to generate X
	    X = self.debugInitializeWeights(m, input_layer_size - 1)
	    y = np.arange(1, 1+m) % num_labels
	    # print(y)
	    # Unroll parameters
	    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

	    # short hand for cost function
	    costFunc = lambda p: self.nnCostFunction(p, input_layer_size, hidden_layer_size,
	                                        num_labels, X, y, lambda_)
	    cost, grad = costFunc(nn_params)
	    numgrad = self.computeNumericalGradient(costFunc, nn_params)

	    # Visually examine the two gradient computations.The two columns you get should be very similar.
	    print(np.stack([numgrad, grad], axis=1))
	    print('The above two columns you get should be very similar.')
	    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

	    # Evaluate the norm of the difference between two the solutions. If you have a correct
	    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
	    # should be less than 1e-9.
	    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

	    print('If your backpropagation implementation is correct, then \n'
	          'the relative difference will be small (less than 1e-9). \n'
	          'Relative Difference: %g' % diff)



	def computeNumericalGradient(self, J, theta, e=1e-4):
	    """
	    Computes the gradient using "finite differences" and gives us a numerical estimate of the
	    gradient.

	    Parameters
	    ----------
	    J : func
	        The cost function which will be used to estimate its numerical gradient.

	    theta : array_like
	        The one dimensional unrolled network parameters. The numerical gradient is computed at
	         those given parameters.

	    e : float (optional)
	        The value to use for epsilon for computing the finite difference.

	    Notes
	    -----
	    The following code implements numerical gradient checking, and
	    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
	    approximation of) the partial derivative of J with respect to the
	    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
	    be the (approximately) the partial derivative of J with respect
	    to theta[i].)
	    """
	    numgrad = np.zeros(theta.shape)
	    perturb = np.diag(e * np.ones(theta.shape))
	    for i in range(theta.size):
	        loss1, _ = J(theta - perturb[:, i])
	        loss2, _ = J(theta + perturb[:, i])
	        numgrad[i] = (loss2 - loss1)/(2*e)
	    return numgrad


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
	        # Display Image
	        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
	                      cmap='Greys', extent=[0, 1, 0, 1])
	        ax.axis('off')


if __name__ == "__main__":
	Exercise1().run()