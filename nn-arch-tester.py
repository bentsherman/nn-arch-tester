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
	net.train(X_train, y_train, num_iter, batch_size, lr, monitor=100)

	return net.evaluate(X_test, y_test)

def kfold(X, y, sizes, num_folds=5):
	X_folds = np.split(X, num_folds)
	y_folds = np.split(y, num_folds)
	scores = []

	for i in xrange(num_folds):
		X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
		y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
		X_test = X_folds[i]
		y_test = y_folds[i]

		scores.append(evaluate(X_train, y_train, X_test, y_test, sizes))

	return scores

# initialize data
X = mnist.train_images
y = to_categorical(mnist.train_labels)

# initialize hyperparameters
layers = [X.shape[1], 10]

# run experiments
scores = kfold(X, y, layers)

print "%.3f +/- %.3f" % (np.mean(scores), np.std(scores))
