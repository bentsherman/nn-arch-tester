import mnist
import network
import numpy as np
import random

def to_categorical(y):
	c = np.unique(y)
	y_cate = np.zeros((y.shape[0], c.shape[0]))
	y_cate[np.arange(y.shape[0]), y] = 1

	return y_cate

def evaluate(X_train, y_train, X_test, y_test, sizes, num_iter=1000, batch_size=100, lr=0.5, monitor=-1):
	net = network.Network(sizes)
	loss = net.train(X_train, y_train, num_iter, batch_size, lr, monitor)

	return net.evaluate(X_test, y_test), loss

def kfold(X, y, sizes, num_folds=5):
	X_folds = np.split(X, num_folds)
	y_folds = np.split(y, num_folds)
	scores = []
	losses = []

	for i in xrange(num_folds):
		# extract train / test sets from folds
		X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
		y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
		X_test = X_folds[i]
		y_test = y_folds[i]

		# evaluate network
		acc, loss = evaluate(X_train, y_train, X_test, y_test, sizes)

		# save test accuracy and loss
		scores.append(acc)
		losses.append(loss)

	return scores, losses

# initialize data
X = mnist.train_images
y = to_categorical(mnist.train_labels)

# initialize architectures
archs = [
	# part 1: number of hidden layers
	[X.shape[1], 10],
	[X.shape[1], 32, 10],
	[X.shape[1], 32, 32, 10],
	[X.shape[1], 32, 32, 32, 10],

	# part 2: size of hidden layer
	[X.shape[1], 16, 10],
	[X.shape[1], 32, 10],
	[X.shape[1], 64, 10],
	[X.shape[1], 128, 10],
	[X.shape[1], 256, 10],

	# part 3: shape of network
	[X.shape[1], 32, 64, 128, 10],
	[X.shape[1], 64, 64, 64, 10],
	[X.shape[1], 128, 64, 32, 10]
]

# run experiments
for arch in archs:
	print "-".join([str(n) for n in arch])

	# compute test accuracy, loss
	scores, losses = kfold(X, y, arch)

	# print average loss time series
	loss = np.mean(losses, axis=0)[::20]

	for i in xrange(len(loss)):
		print "%.3f" % (loss[i])

	# print mean and stddev of accuracy
	print "%.3f +/- %.3f" % (np.mean(scores), np.std(scores))
