"""Module progressions_list.

This module creates a list of objects representing sequences of chords, ie harmonic progressions,
in order to train the embedding space.
The style depends on the scales used and is currently classical + pop + jazz.

Example
-------
How to use this code

	import progressions_list
	prog_litst = progressions_list()

"""

base_major = [60, 62, 64, 65, 67, 69, 71]             # do re mi fa sol la si do
base_minor = [57, 59, 60, 62, 64, 65, 68]             # la si do re mi fa sol# la
base_minor_melodic = [57, 59, 60, 62, 64, 66, 68]     # la si do re mi fa# sol# la
base_pentatonic = [60, 62, 64, 67, 69]                # do re mi sol la do
base_blues = [60, 63, 65, 67, 69, 71]                 # do mib fa fa# sol sib do
base_mixolydian = [55, 57, 59, 60, 62, 64, 65]        # sol la si do re mi fa sol
base_dorian = [62, 64, 65, 67, 69, 71, 72]            # re mi fa sol la si do re
base_eolian = [57, 59, 60, 62, 64, 65, 67]            # la si do re mi fa sol la


class Progression:

	"""This structure will be used to manually describe musical harmonic progressions.
	They are here formalised in an symbolic and synthetic way.

	Attributes
	----------
	base : int list
		One of the scale defined at the beginning of this file.
	chords : dict of int list -> str list
		A dictionnary containing, for the given base, the degrees needed and the type of chord to construct for each degree.
        A degre equal to 0 indicates the first note of the scale (to make it list-friendly).
    name : str
    	To identify the progression

	"""

	def __init__(self, base, chords, name):
		self.base = base
		self.chords = chords
		self.name = name


def progressions_list():

	# Aggregator : Constructs a list of symbolic harmonic progressions.
	# These sequences of chords come from music theory and observation of songs as examples.

	res = []

	res.append(Progression(base_major, [[0,"ap"], [1,"ap"], [4,"ap"]], "M1"))
	res.append(Progression(base_major, [[0,"ap"], [3,"ap"], [4,"ap"]], "M2"))
	res.append(Progression(base_major, [[0,"ap"], [4,"ap"], [3,"ap"], [4,"ap"]], "Swift"))
	res.append(Progression(base_major, [[0,"ap"], [5,"ap"], [4,"ap"], [3,"ap"]], "Foo"))
	res.append(Progression(base_major, [[0,"ap"], [3,"ap"], [4,"ap"], [3,"ap"]], "M3"))
	res.append(Progression(base_major, [[0,"ap"], [3,"ap"], [0,"ap"], [1,"ap"]], "Morissette"))
	res.append(Progression(base_eolian, [[0,"ap"], [3,"ap"], [1,"ap"], [4,"ap"]], "m1"))
	res.append(Progression(base_eolian, [[0,"ap"], [5,"ap"], [2,"ap"], [6,"ap"]], "Perry"))
	res.append(Progression(base_eolian, [[0,"ap"], [6,"ap"], [5,"ap"], [3,"ap"]], "Eiffel"))
	res.append(Progression(base_blues, [[0,"ap"], [3,"7th"], [4,"7th"]], "blues1"))
	res.append(Progression(base_major, [[0,"ap"], [6,"ap"], [5,"ap"], [6,"ap"], [0,"ap"], [6,"ap"], [5,"ap"], [4,"ap"]], "free1"))
	res.append(Progression(base_major, [[0,"ap"], [0,"ap"], [3,"ap"], [4,"ap"], [0,"ap"], [0,"ap"], [6,"ap"], [3,"ap"]], "free2"))
	res.append(Progression(base_major, [[4,"ap"], [1,"ap"], [5,"ap"], [3,"ap"], [4,"ap"], [1,"ap"], [5,"ap"], [0,"ap"]], "free3"))
	res.append(Progression(base_major, [[0,"ap"], [3,"ap"], [0,"ap"], [4,"ap"]], "free4"))
	res.append(Progression(base_major, [[0,"ap"], [5,"ap"], [1,"ap"], [4,"ap"], [0,"ap"], [5,"ap"], [1,"ap"], [4,"ap"]], "free5"))
	res.append(Progression(base_minor, [[0,"ap"], [3,"ap"], [0,"ap"], [3,"ap"], [0,"ap"], [3,"ap"], [5,"ap"], [4,"ap"]], "free6"))
	res.append(Progression(base_minor, [[0,"ap"], [6,"ap"], [3,"ap"], [4,"ap"], [0,"ap"], [6,"ap"], [3,"ap"], [4,"ap"]], "free7"))
	res.append(Progression(base_minor, [[0,"ap"], [4,"ap"], [4,"ap"], [6,"ap"], [0,"ap"], [4,"ap"], [3,"ap"], [4,"ap"]], "free8"))
	res.append(Progression(base_major, [[0,"ap"], [4,"ap"], [4,"ap"], [6,"ap"], [0,"ap"], [4,"ap"], [3,"ap"], [4,"ap"]], "free9"))
	res.append(Progression(base_major, [[0,"ap"], [6,"ap"], [5,"ap"], [4,"ap"], [0,"ap"], [6,"ap"], [5,"ap"], [4,"ap"]], "free10"))
	res.append(Progression(base_major, [[0,"ap"], [0,"ap"], [6,"ap"], [3,"ap"], [0,"ap"], [0,"ap"], [3,"ap"], [4,"ap"]], "free11"))
	res.append(Progression(base_minor, [[0,"ap"], [2,"ap"], [3,"ap"], [4,"ap"], [0,"ap"], [2,"ap"], [3,"ap"], [4,"ap"]], "free12"))
	res.append(Progression(base_major, [[0,"ap"], [5,"ap"], [3,"ap"], [4,"ap"], [0,"ap"], [5,"ap"], [3,"ap"], [4,"ap"]], "free13"))
	return res




