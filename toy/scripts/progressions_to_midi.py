"""Module generate_progressions.

This module creates .mid files representing sequences of chords, ie harmonic progressions,
in order to train the embedding space.
The style depends on the scales used and is currently classical + pop + jazz.
The output is written to the specified directory (by default, 'toy/dataset/progressions').

Each sequence contains nb_notes chords, including one for labeling.
Transpositions and circular permutations enhance the dataset size.

Example
-------
How to use this code

	$ python generate_progressions.py

"""

from midiutil.MidiFile import MIDIFile
import numpy as np
import itertools


# tempo : 60 bpm
global_tempo = 60
# full sequences : 10 chords + 1 label
nb_chords = 11


def generate_midi_progression(pitches, output_name):
	"""Generates .mid chord sequences at given pitches.
	Writes the ouptut to midi files.

	Parameters
	----------
	pitches : int list list
			For each time step, a list of the Midi pitches of the corresponding chord.
	output_name : str
			A name for this harmonic progression's file.

	Returns
	-------
			None.
			Writes generated sequence to disk.

	"""

	# create the MIDI object
	mf = MIDIFile(1)     # only 1 track
	track = 0   		 # the only track
	channel = 0

	time = 0    		 # start at the beginning of the track on beat 0
	mf.addTrackName(track, time, "Sample Track")
	mf.addTempo(track, time, global_tempo)

	for chord in pitches:

		for note in chord:

			# add the note
			# 1 is the duration of the note
			# 100 is the volume
			mf.addNote(track, channel, note, time, 1, 100)

		# all chords last one beat
		time = time + 1

	# finally, write sequence to disk
	with open(output_name, 'wb') as outf:
		mf.writeFile(outf)


def generate_all_midi_progressions(progressions_list, folder):
	"""Generates parameters for generate_midi_progression by.

	Parameters
	----------
	progressions_list : Progression list
			Progressions are custom data structures described in the file progressions_list.py.
	folder : str
			The base name of the folder where to write the midi files.

	Returns
	-------
	None
	"""

	for prog in progressions_list:

		nb_permut = len(prog.chords)
		prog_name = prog.name
		len_base = len(prog.base)

		for p in range(nb_permut):

			for transpo in range(12):

				# Adding the number of the permutation and of the transposition
				# to the file name
				output_name = prog_name + '_p' + \
					str(p) + '_t' + str(transpo) + '.mid'
				pitches = []

				for i in prog.chords:

					# Put together the midi notes required for this chord
					degree = i[0]
					chord_type = i[1]
					
					low = prog.base[degree] + transpo
					d2 = degree + 2
					low2 = prog.base[d2 % len_base] + \
						transpo + (d2 // len_base) * 12
					d4 = degree + 4
					low4 = prog.base[d4 % len_base] + \
						transpo + (d4 // len_base) * 12
					d6 = degree + 6
					low6 = prog.base[d6 % len_base] + \
						transpo + (d6 // len_base) * 12

					if chord_type == 'ap':
						pitches.append([low, low2, low4])
					elif chord_type == '7th':
						pitches.append([low, low2, low4, low6])

				pitches = pitches * \
					int((nb_chords + nb_permut) / nb_permut + 1)
				# the first chords represent the data
				pitches_data = pitches[p:p + nb_chords - 1]
				# the last one is the label
				pitches_label = [pitches[p + nb_chords - 1]]
				generate_midi_progression(
					pitches_data, folder + 'x' + output_name)
				generate_midi_progression(
					pitches_label, folder + 'y' + output_name)


