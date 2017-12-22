"""Module midi_to_data.

This module helps creating trainable data from our midi files.
It reads the data and stores them in numpy arrays.

Example
-------
How to use this code

	import midi_to_data as md
	(X, y, Xval, yval) = md.load_midi_prediction('toy/dataset/progressions/')

"""

import numpy as np
import math
import os
from keras.utils import np_utils
from music21 import converter
from matplotlib import pyplot as plt

# Utility functions used by importMIDI

def get_start_time(el, measure_offset, quantization):
	if (el.offset is not None) and (el.measureNumber in measure_offset):
		return int(math.ceil(
			((measure_offset[el.measureNumber] or 0) + el.offset) * quantization))


def get_end_time(el, measure_offset, quantization):
	if (el.offset is not None) and (el.measureNumber in measure_offset):
		return int(math.ceil(
			((measure_offset[el.measureNumber] or 0) + el.offset + el.duration.quarterLength) * quantization))


def get_pianoroll_part(part, quantization):
	# Gets the measure offsets
	measure_offset = {None: 0}

	# Gets the duration of the part
	duration_max = 0
	for el in part.elements:
		t_end = get_end_time(el, measure_offset, quantization)
		if(t_end > duration_max):
			duration_max = t_end
	# Gets the pitch and offset+duration
	piano_roll_part = np.zeros((128, int(math.ceil(duration_max))))

	n_el = len(part.elements)
	pe = part.elements
	for i in range(1, n_el):
		this_chord = pe[i]
		note_start = get_start_time(this_chord, measure_offset, quantization)
		note_end = get_end_time(this_chord, measure_offset, quantization)
		for this_note in this_chord.pitches:
			piano_roll_part[this_note.midi, note_start:note_end] = 1
	return piano_roll_part


def importMIDI(f):
	# The important function that is used to read the midi files
	piece = converter.parse(f)
	all_parts = {}
	for part in piece.parts:
		try:
			track_name = part[0].bestName()
		except AttributeError:
			track_name = 'None'
		cur_part = get_pianoroll_part(part, 16)
		if (cur_part.shape[1] > 0):
			all_parts[track_name] = cur_part
	return all_parts


def load_midi_prediction(directory):
	""" Loads the midi files and creates a trainable dataset for prediction.
	This may take a couple of seconds to complete.

	Parameters
	----------
	directory : str
	  The location of the directory containing the midi files.

	Returns
	-------
	Xtrain, ytrain, Xval, yval : np arrays
	  The training and validation data for midi chords prediction on this file.
	  X contains an array of 10 chords, y is the 11th chord to predict.
	  These arrays contain, for each chord, a 128-long list indicating the activation of each midi note.

	"""

	filenames = os.listdir(directory)
	filenames = [x[1:] for x in filenames]
	filenames = list(set(filenames))
	n_files = len(filenames)

	dataX = []
	dataY = []

	# prepare to split the dataset into training and validation data
	ratio = 0.7
	idx_train = int(ratio * n_files)

	# 1 midi event = 16 channels -> Keep 1 column out of 16
	columns_to_keep = [16 * i for i in range(10)]

	for f in filenames:
		xfile = directory + 'x' + f
		yfile = directory + 'y' + f
		Xall_parts = importMIDI(xfile)
		Yall_parts = importMIDI(yfile)
		Xall_parts = Xall_parts['None'][:, columns_to_keep]
		Yall_parts = Yall_parts['None'][:, 0]
		dataX.append(Xall_parts)
		dataY.append(Yall_parts)

	print("Midi data loaded.")

	dataXtrain = dataX[:idx_train]
	print(len(dataXtrain), 'training samples')
	dataXval = dataX[idx_train:]
	print(len(dataXval), 'validation samples')
	dataYtrain = dataY[:idx_train]
	dataYval = dataY[idx_train:]

	# reshape data to be [samples, time steps, features] expected by an LSTM network.
	# Here (samples, 10, 128) and (samples, 128)

	Xtrain = np.array([np.transpose(x) for x in dataXtrain])
	Xval = np.array([np.transpose(x) for x in dataXval])
	ytrain = np.array(dataYtrain)
	yval = np.array(dataYval)

	return Xtrain, ytrain, Xval, yval
