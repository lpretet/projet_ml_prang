"""Some functions to customize the training process.

Example
-------
How to use this code

	import training_helpers as tr
	from keras.models import Sequential
	model = Sequential()
	model.compile(loss='mse', optimizer='rmsprop', metrics=[tr.frame_loss])

"""

import keras.backend as K
import numpy as np 
import tensorflow as tf

def frame_loss(y_true, y_pred):
	""" A frame-wise loss measure more suitable to sparse data like Midi vectors.
	It is inspired from the definition of accuracy in :
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.205.9184&rep=rep1&type=pdf

	It is not a trainable measure, since it is not differentiable, but it can be used as an indicator of progres.

	Parameters
	----------
	y_true : tensor
		The expected output of the network.
	y_pred : tensor
		The actual output of the network.
		
	Returns
	-------
	score : single value tensor
		The frame_wise loss between the two inputs.

	"""
	
	threshold = 0.5
	y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
	falses = K.sum(K.abs(y_true - y_pred))
	true_positives = K.maximum(K.sum(y_pred*y_true), 0.1) # To avoid division by 0
	score = (falses+true_positives)/true_positives
	return score

