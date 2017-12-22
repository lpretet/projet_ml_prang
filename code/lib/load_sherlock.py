"""Module load_sherlock.

This module creates offers a service to load the text dataset of Sherlock Holme's stories stored in a text file.
It is used in Exercise 2 to compare the performances of different RNNs for text prediction.

Example
-------
How to use this code

	import load_sherlock as sh

"""

# The code framework to load the data was taken at :
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
from keras.utils import np_utils


def load():
	""" Loads the text file and creates a trainable dataset from it.

	Parameters
	----------
	filename : str
					The location of Sherlock's text file.

	Returns
	-------
	Xtrain, ytrain, Xval, yval : np arrays
					The training and validation data for text prediction on this file.

	"""

	# load ascii text and covert to lowercase
	filename = "code/lib/sherlock.txt"
	raw_text = open(filename).read()
	raw_text = raw_text.lower()
	n_chars = len(raw_text)
	# create mapping of unique chars to integers
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)
	# prepare to split the dataset into training and validation data
	ratio = 0.7
	idx_train = int(ratio * n_chars)
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print("Total Patterns: ", n_patterns)
	# now spilt into training and validation data
	dataXtrain = dataX[:idx_train]
	dataXval = dataX[idx_train:]
	dataYtrain = dataY[:idx_train]
	dataYval = dataY[idx_train:]
	# reshape data to be [samples, time steps, features] expected by an LSTM
	# network.
	Xtrain = np.reshape(dataXtrain, (len(dataXtrain), seq_length, 1))
	Xval = np.reshape(dataXval, (len(dataXval), seq_length, 1))
	# normalize
	Xtrain = Xtrain / float(n_vocab)
	Xval = Xval / float(n_vocab)
	# ONE HOT encoding for the OUPTUT variable
	ytrain = np_utils.to_categorical(dataYtrain)
	yval = np_utils.to_categorical(dataYval)

	return Xtrain, ytrain, Xval, yval
