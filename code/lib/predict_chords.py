"""Module predict_chords.

Here we laod a pre-trained model, select randomly a chord sequence of chords, and have it
predict the last chords. Then we write the output back to Midi to compare the expected and predicted
chords by hearing.

Example
-------
How to use this code

	-- script.py --
	import predict_chords as pr
	model_path = 'code/models/MidiCNNModel2.h5'
	test_idx = 68
	pr.predict_and_write(test_idx, model_path, 'test')

	-- shell --
	$ python script.py
	$ timidity test_pred.mid
	$ timidity test_real.mid

"""


import keras.models
import numpy as np
from midiutil.MidiFile import MIDIFile
import midi_to_data as md
import training_helpers as tr


def predict_and_write(test_idx, model_path, output_path):
	"""Generates the .mid chord predicted by the designated model for a certain input.
	Writes the ouptut to a midi file.

	Parameters
	----------
	test_idx : int
			The index of the sequence to use for prediction of which we want to predict the last chord.
			It will be taken from Xval so the index must not exceed this array's dimension.
	model_path : str
			The path where to find the serialized model (already trained).
	output_path : str
			The path where to write the generated midi file.


	Returns
	-------
	None.
		Writes generated sequence to disk.

	"""

	[X, y, Xval, yval] = md.load_midi_prediction('toy/dataset/progressions/')
	model = keras.models.load_model(
		model_path, custom_objects={
			'frame_loss': tr.frame_loss})
	x = Xval[test_idx]
	y = np.array(yval[test_idx])
	x = np.reshape(x, (1, 10, 128))

	y_predicted = model.predict(x)[0]
	y_predicted = np.array([int(y > 0.5) for y in y_predicted])
	print(y_predicted)

	data_to_midi(y_predicted, output_path + '_pred' + '.mid')
	data_to_midi(y, output_path + '_real' + '.mid')


def data_to_midi(data, output_name):
	"""Generates .mid chord sequences corresponding to the given array.
	Writes the ouptut to midi files.

	Parameters
	----------
		data : numpy array
			For each time step, a list of the Midi pitches of the corresponding chord.
		output_name : str
			A name for this chord's file.

	Returns
	-------
		None.
		Writes generated sequence to disk.

	"""

	# create the MIDI object
	mf = MIDIFile(1)     # only 1 track
	track = 0   		 # the only track
	channel = 0
	global_tempo = 60

	time = 0    		 # start at the beginning of the track on beat 0
	mf.addTrackName(track, time, "Sample Track")
	mf.addTempo(track, time, global_tempo)

	for idx, val in enumerate(data):

		if val == 1:
			# add the note
			# 1 is the duration of the note
			# 100 is the volume
			mf.addNote(track, channel, idx, time, 1, 100)

	# finally, write sequence to disk
	with open(output_name, 'wb') as outf:
		mf.writeFile(outf)
