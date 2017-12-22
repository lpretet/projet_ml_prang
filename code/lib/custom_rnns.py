"""Classes of RNNs and LSTMs redefined by ourselves.
These classes serve for comparison with the native Keras version in Exercise 2.

Example
-------
How to use this code

	from custom_rnns import MinimalLSTMCell, MinimalRNNCell

"""

import keras
from keras.engine.topology import Layer
from keras import backend as K


class MinimalRNNCell(keras.layers.Layer):
	""" Basic RNN Cell.

	Attributes
	----------
	units : int
		The number of hidden units in the layer.
	state_size : int
		In such a  single state RNN, it is the size of the cell output.

	Example
	-------
	How to use this class

			model.add(RNN(MinimalRNNCell(256), input_shape=(X.shape[1], X.shape[2])))

	"""

	def __init__(self, units, **kwargs):

		self.units = units
		self.state_size = units
		super(MinimalRNNCell, self).__init__(**kwargs)

	def build(self, input_shape):

		# This layer is composed of two sets of weights : forward and
		# recurrent.
		self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
									  initializer='uniform',
									  name='kernel')
		self.recurrent_kernel = self.add_weight(
			shape=(self.units, self.units),
			initializer='uniform',
			name='recurrent_kernel')
		self.built = True

	def call(self, inputs, states):
		# Get the state of the previous iteration
		prev_output = states[0]
		# First apply the forward kernel
		h = K.dot(inputs, self.kernel)
		# Then the recurrent kernel.
		output = h + K.dot(prev_output, self.recurrent_kernel)
		return output, [output]


class MinimalLSTMCell(keras.layers.Layer):
	""" Basic LSTM Cell.

	Attributes
	----------
	units : int
		The number of hidden units in the layer.
	state_size : (int, int)
		The sizes of the outputs [h, c] of both states

	Example
	-------
	How to use this class

			model.add(RNN(MinimalLSTMCell(256), input_shape=(X.shape[1], X.shape[2])))

	"""

	def __init__(self, units, **kwargs):

		super(MinimalLSTMCell, self).__init__(**kwargs)
		self.units = units
		self.state_size = (self.units, self.units)

	def build(self, input_shape):

		input_dim = input_shape[-1]

		# The usual forward kernel
		self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
									  name='kernel',
									  initializer='uniform')
		# The usual recurrent kernel
		self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
												name='recurrent_kernel',
												initializer='uniform')
		# The gate's kernels : input, forget, candidate, output
		self.kernel_i = self.kernel[:, :self.units]
		self.kernel_f = self.kernel[:, self.units: self.units * 2]
		self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
		self.kernel_o = self.kernel[:, self.units * 3:]
		# The associated recurrent kernels
		self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
		self.recurrent_kernel_f = self.recurrent_kernel[:,
														self.units: self.units * 2]
		self.recurrent_kernel_c = self.recurrent_kernel[:,
														self.units * 2: self.units * 3]
		self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

		self.built = True

	def call(self, inputs, states):

		h_tm1 = states[0]  # previous memory state
		c_tm1 = states[1]  # previous carry state

		x_i = K.dot(inputs, self.kernel_i)
		x_f = K.dot(inputs, self.kernel_f)
		x_c = K.dot(inputs, self.kernel_c)
		x_o = K.dot(inputs, self.kernel_o)

		i = K.hard_sigmoid(x_i + K.dot(h_tm1, self.recurrent_kernel_i))
		f = K.hard_sigmoid(x_f + K.dot(h_tm1, self.recurrent_kernel_f))
		c = f * c_tm1 + i * K.tanh(x_c + K.dot(h_tm1, self.recurrent_kernel_c))
		o = K.hard_sigmoid(x_o + K.dot(h_tm1, self.recurrent_kernel_o))

		h = o * K.tanh(c)

		return h, [h, c]
