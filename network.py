import numpy as np
import random

class Network(object):

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.weights = [np.random.randn(n, k)/np.sqrt(k) for (k, n) in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(n, 1) for n in sizes[1:]]

	def feedforward(self, a):
		a = a.reshape(a.shape[0], 1)

		for W, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(W, a) + b)

		return a

	def backprop(self, x, y):
		x = x.reshape(x.shape[0], 1)
		y = y.reshape(y.shape[0], 1)

		delta_W = [None for W in self.weights]
		delta_b = [None for b in self.biases]

		# forward pass
		A = [x]
		Z = []
		a = x

		for W, b in zip(self.weights, self.biases):
			z = np.dot(W, a) + b
			Z.append(z)

			a = sigmoid(z)
			A.append(a)

		# accumulate cost
		self._cost[-1] += self.cost(A[-1], y)

		# backward pass
		delta = self.cost_deriv(A[-1], y)

		delta_W[-1] = np.dot(delta, A[-2].T)
		delta_b[-1] = delta

		for l in xrange(2, self.num_layers):
			delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_deriv(Z[-l])
			delta_W[-l] = np.dot(delta, A[-l-1].T)
			delta_b[-l] = delta

		return (delta_W, delta_b)

	def cost(self, a, y):
		return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

	def cost_deriv(self, a, y):
		return a - y;

	def train(self, X_train, y_train, num_iter, batch_size, lr, monitor=-1):
		n = X_train.shape[0]
		self._cost = []

		for t in xrange(num_iter):
			# initialize cost
			self._cost.append(0)

			# sample mini-batch from training set
			indices = random.sample(xrange(n), batch_size)
			x_batch = X_train[indices]
			y_batch = y_train[indices]

			# compute parameter updates over mini-batch
			delta_W = [np.zeros(W.shape) for W in self.weights]
			delta_b = [np.zeros(b.shape) for b in self.biases]

			for (x, y) in zip(x_batch, y_batch):
				dW, db = self.backprop(x, y)

				delta_W = [dW + dW_i for dW, dW_i in zip(delta_W, dW)]
				delta_b = [db + db_i for db, db_i in zip(delta_b, db)]

			# update weights
			self.weights = [W - lr / batch_size * dW for W, dW in zip(self.weights, delta_W)]
			self.biases = [b - lr / batch_size * db for b, db in zip(self.biases, delta_b)]

			if monitor != -1 and (t % monitor == 0):
				print "%d %0.2f" % (t, self.evaluate(x_batch, y_batch))

		return self._cost

	def evaluate(self, X_test, y_test):
		y = [self.feedforward(x) for x in X_test]

		return sum([int(np.argmax(y[i]) == np.argmax(y_test[i])) for i in xrange(len(y))]) / float(len(y))

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))
