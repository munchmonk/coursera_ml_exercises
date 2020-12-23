#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Regularized logistic regression (to avoid overfitting)
"""

import numpy as np 
from matplotlib import pyplot
from scipy import optimize
import math

data = np.loadtxt('data2.txt', delimiter=',')
# X -> result of two different tests run on a microchip
# y -> whether the microchip passed QA or not
X = data[:, :2]
y = data[:, 2]


def plotData(X, y):
	"""
	Plots the data to get an initial visualization. The two axes represent the two column of X (i.e. the results of the two tests),
	and the output y is indicated with a * for a positive example (y=1) and o otherwise (y=0)
	"""

	fig = pyplot.figure()

	# Store the indices where y is positive or negative. pos and neg are both lists with the same length as y containing boolean values
	pos = y == 1
	neg = y == 0

	# Plots only the training examples with positive result
	# k* -> black star, lw -> line width, ms -> marker size
	pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)

	# Plots only the training examples with negative result
	# ko -> black circle, mfc -> marker face color (yellow), ms -> marker size, mec -> marker edge color (black), mew -> marker edge width
	pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms='8', mec='k', mew=1)

	pyplot.xlabel('Microchip test 1')
	pyplot.ylabel('Microchip test 2')
	pyplot.legend(['y = 1', 'y = 0'], loc='upper right')

	# pyplot.show()


# # -----------------------------------
# # We plot the data and see that a normal application of logistic regression won't work as the dataset cannot be separated by a 
# # straight line - and logistic regression only finds a decision boundary
# plotData(X, y)


def mapFeature(X1, X2, degree=6):
    """


    ---- N.B. I did not write this function, I just copied it from utils.py ----




    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
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


# -----------------------------------
# We use this provided function to map the two features into a higher order polynomial (we use all x1 and x2 terms up to the sixth power)
# The function also adds a column of 1 at the beginning.
# Thid way we will have a more complex, non-linear decision boundary that can better capture our dataset. This however could create overfitting problems
# which we will tackle next.
X = mapFeature(X[:, 0], X[:, 1])


def computeSigmoid(x):
	"""
	Helper function to avoid having to rewrite this computation several times
	"""
	return 1 / (1 + math.exp(-x))


def sigmoid(z):
	"""
	Compute the sigmod function on imput z. g(z) = 1 / (1 + e^-z)

	Note: g(z) -> 0 for z -> -inf, g(z) -> 1 for z -> +inf and g(0) = 0.5

	z is an array_like element and it can be a scalar, a vector or a matrix. The function should perform the sigmoid function on every element if z is
	a vector or a matrix.

	The function returns g, an array_like element of the same shape as z (since, again, the sigmoid function is performed element-wise)
	"""

	# Convert z to a numpy array
	z = np.array(z)

	# Return variable
	g = np.zeros(z.shape)

	# 0: scalar, 1: vector, 2: matrix
	dim = np.ndim(z)

	if dim == 0:
		g = np.array(computeSigmoid(z))

	elif dim == 1:
		for i in range(len(z)):
			g[i] = computeSigmoid(z[i])

	elif dim == 2:
		for i in range(z.shape[0]):
			for j in range(z.shape[1]):
				g[i][j] = computeSigmoid(z[i][j])

	return g


def costFunctionReg(theta, X, y, lambda_):
	"""
	Compute cost and gradient for logistic regression with regularization.

	theta: vector of length n containing the logistic regression parameters
	X: m x n matrix containing the dataset
	y: vector of length m containing the data labels
	m: number of examples
	n: number of features
	lambda_: float, the regularization parameter

	Returns:
	J: float, the computed value for the cost function
	grad: vector of length n containing the gradient of the cost function with respect to theta, calculated for the current theta
	"""

	m = y.size
	J = 0
	grad = np.zeros(theta.shape)

	n = theta.size

	# -----------------------------------
	# Compute cost
	# ref. ex_1.py for more explanations 
	# First sum
	for i in range(m):
		h = sigmoid(np.dot(theta.T, X[i]))
		J += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
	J /= m

	# Second sum (note how we do not include theta[0] in the calculation so as not to "penalize" it
	tmp = 0
	for i in range(1, n):
		tmp += theta[i] ** 2
	tmp *= lambda_ / (2 * m)
	J += tmp

	# -----------------------------------
	# Compute gradient
	for j in range(n):
		for i in range(m):
			h = sigmoid(np.dot(theta.T, X[i]))

			grad[j] += (h - y[i]) * X[i][j]

		grad[j] /= m

		# Do not regularize theta[0]
		if j > 0:
			grad[j] += lambda_ * theta[j] / m


	return J, grad


# # Try some test values
# initial_theta = np.zeros(X.shape[1])
# lambda_ = 1
# cost, grad = costFunctionReg(initial_theta, X, y, lambda_)
# print(cost)
# print(grad)

# initial_theta = np.ones(X.shape[1])
# lambda_ = 10
# cost, grad = costFunctionReg(initial_theta, X, y, lambda_)
# print(cost)
# print(grad)


# -----------------------------------
# We use scipy.optimize.minimize to calculate an optimal theta instead of implementing our own gradient descent function (ref. ex_1.py)
initial_theta = np.zeros(X.shape[1])
options = {'maxiter': 100}
lambda_ = 1
res = optimize.minimize(costFunctionReg, initial_theta, (X, y, lambda_), jac=True, method='TNC', options=options)
cost = res.fun
theta = res.x


def plotDecisionBoundary(plotData, theta, X, y):
    """


    ---- N.B. I did not write this function, I merely copied and pasted it from the assignment. It was given in utils.py ----


    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
    	# ---- Comment added by me:
    	# ---- the decision boundary is given by Theta0 + x1*Theta1 + x2*Theta2 = 0
    	# ---- if we treat x1 as x and x2 as y we get y = -(1/Theta2)*(Theta0+Theta1*x) which is a straight line
    	# ---- so we take any two points (in this case they chose somewhere before the smallest result of the first exam and somewhere after the biggest),
    	# ---- we put them into the decision boundary we found to obtain two points and use them to plot the whole straight line
    	# ---- Note: the results of the fist exam are in what is now the second column of our matrix X (as the first column is all ones)

        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)


# Plot our training examples together with the decision boundary (which we obtained by finding a value of theta that minimizes J, our cost function)
plotDecisionBoundary(plotData, theta, X, y)
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)
pyplot.show()


# -----------------------------------
# We chose the parameter lambda_ somewhat arbitrarily; graphically, we understand that too big a lambda_ underfits our dataset, and too small a lambda_
# overfits it. We want to calculate the training accuracy against our dataset using the theta we calculated (which in turn depends on the lambda_ we chose!)


def predict(theta, X):
	"""
	This function tests the prediction model we calculated against the dataset it was fed to see how accurately it predicts values we
	already know (i.e. how well the model works).

	theta is a vector of length n+1
	X is the training example matrix of dimensions m x n+1 (the first column is all ones)

	The return value is p, an array of length m containing the prediction for each entry (either 0 or 1).

	We assign 1 to each value for which h_theta(x) >= 0.5 and 0 otherwise.

	N.B. the sigmoid function always returns a real value between 0 and 1.
	"""

	m = X.shape[0]

	p = np.zeros(m)

	for i in range(m):
		prob = sigmoid(np.dot(theta.T, X[i]))
		p[i] = 1 if prob >= 0.5 else 0

	return p


p = predict(theta, X)
training_accuracy = np.mean(p == y) * 100
print('Training accuracy: {:.1f}%'.format(training_accuracy))



















