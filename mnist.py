import numpy as np
import operator
import struct as st

def prod(iterable):
	return reduce(operator.mul, iterable, 1)

def load_idx(filename, rank):
	f = open(filename, "rb")

	magic = st.unpack(">4B", f.read(4))
	num_idx = magic[3]
	sizes = [st.unpack(">I", f.read(4))[0] for i in xrange(num_idx)]

	num_bytes = prod(sizes)
	n = sizes[0]
	m = prod(sizes[1:])

	shape = sizes[0:rank-1] + [prod(sizes[rank-1:])]

	return np.asarray(st.unpack(">%dB" % (num_bytes), f.read(num_bytes))).reshape(shape)

train_images = load_idx("train-images-idx3-ubyte", 2) / 255.0
train_labels = load_idx("train-labels-idx1-ubyte", 1)

test_images = load_idx("t10k-images-idx3-ubyte", 2) / 255.0
test_labels = load_idx("t10k-labels-idx1-ubyte", 1)
