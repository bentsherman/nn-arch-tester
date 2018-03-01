import mnist
import network
import numpy as np
import random

def to_categorical(y):
	c = np.unique(y)
	y_cate = np.zeros((y.shape[0], c.shape[0]))
	y_cate[np.arange(y.shape[0]), y] = 1

	return y_cate

def evaluate(X_train, y_train, X_test, y_test, sizes, num_iter=1000, batch_size=100, lr=0.5):
	net = network.Network(sizes)
	net.train(X_train, y_train, num_iter, batch_size, lr)

	return net.evaluate(X_test, y_test)

def kfold(X, y, sizes, num_folds=5):
	n = X.shape[0]
	n_train = n * (num_folds - 1) / num_folds
	n_test = n - n_train
	indices = random.sample(xrange(n), n)
	scores = []

	for i in xrange(num_folds):
		X_train = X[indices[0:n_train]]
		y_train = y[indices[0:n_train]]
		X_test = X[indices[n_train:]]
		y_test = y[indices[n_train:]]

		scores.append(evaluate(X_train, y_train, X_test, y_test, sizes))

	return scores

# initialize data
X_train = mnist.train_images
y_train = to_categorical(mnist.train_labels)

X_test = mnist.test_images
y_test = to_categorical(mnist.test_labels)

# initialize hyperparameters
layers = [X_train.shape[1], 10]

# run experiments
scores = kfold(X_train, y_train, layers)

print "%.3f +/- %.3f" % (np.mean(scores), np.std(scores))
