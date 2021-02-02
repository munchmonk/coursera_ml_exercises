#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
	Logistic regression (i.e. classification problems)
"""

import numpy as np
from matplotlib import pyplot
from scipy import optimize
import math

data = np.loadtxt('data1.txt', delimiter=',')
# X -> the marks (results) of two different exams taken by the same student
# y -> whether said student was admitted into university or not
X, y = data[:, :2], data[:, 2]


def plotData(X, y):
	"""
	Plots the data to get an initial visualization. The two axes represent the two column of X (i.e. the results of the two exams),
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

	pyplot.xlabel('Results of first exam')
	pyplot.ylabel('Results of second exam')
	pyplot.legend(['Admitted', 'Not admitted'])

	# pyplot.show()


# plotData(X, y)


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


# # -----------------------------------
# Testing out different inputs for sigmoid(z)
# a = 0
# print(sigmoid(a))
# a = [0.5, 1]
# print(sigmoid(a))
# a = [[-100, 0, 2, 300], [0, 1, 2, 3]]
# print(sigmoid(a))


# -----------------------------------
# m = number of training examples (rows)
# n = number of features (columns)
m, n = X.shape

# Add the intercept term to X (i.e. append a column of ones to the beginning of X)
X = np.concatenate([np.ones((m, 1)), X], axis=1)


def costFunction(theta, X, y):
	"""
	Compute cost and gradient for logistic regression.

	theta: (n+1) length vector containing the parameters for logistic regression

	X: (m x n+1) matrix containing training examples

	y: m length vector containing the results of the training examples (0 or 1) aka labels for the input

	The function returns J and grad:

	J: float, the computed value of the cost function

	grad: (n+1) length vector, gradient of the cost function with respect to theta for the current values of theta

	"""

	n = theta.size - 1
	m = y.size

	J = 0
	grad = np.zeros(theta.shape)

	# -----------------------------------
	# Compute cost
	for i in range(m):
		# Note: for logistic regression, h_theta_i(x) = sigmoid(theta.T * X[i])
		h = sigmoid(np.dot(theta.T, X[i]))

		J += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)

	J /= m

	# -----------------------------------
	# Compute gradient
	# Note: the two for loops (0..m) could be combined to save computation time; the yare kept separate here for learning and clarity purposes
	# as this way it's easier to see what we are doing
	for j in range(n+1):
		for i in range(m):
			h = sigmoid(np.dot(theta.T, X[i]))

			grad[j] += (h - y[i]) * X[i][j]

		grad[j] /= m

	return J, grad

# # -----------------------------------
# # testing some example values
# theta = np.zeros(n+1)
# print(costFunction(theta, X, y))
# theta = np.array([-24, 0.2, 0.2])
# print(costFunction(theta, X, y))


# -----------------------------------
# Instead of writing a function to calculate gradient descent like we did in the first exercise (ex_1, ex_1) we use the scipy library to do it for us
# the scipy.optimize module contains the function minimize that is going to minimize our cost function J(theta) by finding appropriate values of theta.

initial_theta = np.zeros(n+1)
options = {'maxiter': 400}

# costFunction is the function we want to minimize, which we wrote earlier, starting from parameters initial_theta
# jac indicates whether the function we provided also outputs the gradient (Jacobian), which is the case here
# TNC is the truncated Newton algorithm. This function returns an OptimizeResult object
res = optimize.minimize(costFunction, initial_theta, (X, y), jac=True, method='TNC', options=options)

# fun: the value of cost function for the optimized theta found. Property of OptimizeResult
cost = res.fun

# x: the optimized parameter (which we called theta) that minimizes the function we provided
theta = res.x

# print('Cost: ', cost)
# print('Theta; ', theta)


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
pyplot.show()


# # -----------------------------------
# # Now that we have theta, to calculate the probability of a new student getting admitted we can use our model:
# # h = sigmoid(theta*X)
# # for example, if their votes were 45 and 85:
# prob = sigmoid(np.dot(theta.T, [1, 45, 85]))
# print(prob)

# -----------------------------------
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
print('Training accuracy: {:.2f}%'.format(training_accuracy))





















