import keras
import numpy as np
import midi_to_data as md
import os

from keras import models
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_midi_simple_visualisation(directory, patterns):
	""" Loads the midi files and creates a trainable dataset for prediction

	Parameters
	----------
	directory : str
	  The location of the directory containing the midi files.
	patterns : str list
		The identification strings contained in the name of the subset of the data to visualize.
		Examples : to visualize only the Eiffel65 progression, patterns = ['Eiffel'].
		To visualize only the Eiffel65 progression with no permutations, patterns = ['Eiffel', 'p0']
	
	Returns
	-------
	X : np array
	  X contains an array of the 10 chords of the given progression.
	  This array contains, for each chord, a 128-long list indicating the activation of each midi note.

	"""

	filenames = os.listdir(directory)
	# Get only the files starting by "x" (data)
	filenames = [directory + x for x in filenames if x[0] == 'x']
	n_files = len(filenames)

	dataX = []
	labels = ['blue'] * n_files
	sizes = [80]*n_files

	# 1 midi event = 16 channels -> Keep 1 column out of 16
	columns_to_keep = [16 * i for i in range(10)]

	for i, f in enumerate(filenames):
		Xall_parts = md.importMIDI(f)
		Xall_parts = Xall_parts['None'][:, columns_to_keep]
		dataX.append(Xall_parts)
		# Get the substructure we want to put forward
		if all(pat in f for pat in patterns):
			labels[i] = 'red'
			sizes[i] = 400

	print("Midi data loaded.")

	# reshape data to be [samples, time steps, features] expected by an LSTM network.
	# Here (samples, 10, 128)
	X = np.array([np.transpose(x) for x in dataX])

	return X, labels, sizes


def visualize_model_pattern(X, labels, sizes, model):
	""" A tool to visualize given points in the embedding space.
	A T-SNE algorithm is used to reduce the dimensionality of the data to be plotted.

	Parameters
	----------
	model : str
		The path to the file containing the serialized version of the model that defines the embedding space.
	X : np array
	  The data in high dimension. The shape must correspond to the input shape of the model.
	labels : str list
		A list that will be used to set the colors of the datapoints.
		A subset of attention can be defined by setting its color in 'labels' to a contrasting one.
	sizes : int list
		Plays the same role as labels to enhance the visibility of the desired subset of points.
	
	Returns
	-------
	None.
		Plots a matplotlib.pyplot figure.

	"""

	# We define here a function that will perform a forward pass of the data through the model
	get_embedded = K.function([model.layers[0].input, K.learning_phase()],
	                                  [model.layers[6].output])

	# Each element of embedded_chords is a 10*N numpy array representing each N embedding features
	# of the 10 chords from the progression. Here N = 10 too.

	# [X, 0] means that we want the output of the network for data X in test mode (no dropout).
	# [X, 1] means training mode.
	embedded_chords = get_embedded([X, 0])[0]

	# Get only the first chord of each progression for visualisation
	chords_to_visualize = np.array([sequence[0] for sequence in embedded_chords])

	print("TSNE performing ...")
	# Random state for TSNE.
	RS = 20150101
	chords_proj = TSNE(3, random_state=RS).fit_transform(chords_to_visualize)
	print("... TSNE performed.")

	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	ax.scatter(chords_proj[:,0], chords_proj[:,1], chords_proj[:,2], sizes=sizes, c=labels, cmap=plt.cm.spectral, edgecolor='k', alpha = 0.3)
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	plt.show()


