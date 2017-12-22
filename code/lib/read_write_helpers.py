"""This module aims at helping saving and loading Python objects.
It is typically used to save training history and avoid re-training the network each time.

Example
-------
How to use this code

	import read_write_helpers as rw

"""

import pickle


def save(obj, name):
	"""Saves training history (dict) and other python objects, serialized.
	Writes '.pickle' files to disk.

	Parameters
	----------
	obj : any type of python object
		The object to serialize
	name : str
		The name of the file to write to.

	Returns
	-------
		None.
		Writes generated pickle to disk.

	"""
	filename = open("code/models/" + name + ".pickle", "wb")
	pickle.dump(obj, filename)
	filename.close()


def load_pickled(filename):
	"""Loads serialized python objects from ".pickle" files.

	Parameters
	----------
	name : str
		The name of the file to read from.

	Returns
	-------
	obj : The deserialized python object.

	"""

	with open(filename, 'rb') as pickle_file:
		obj = pickle.load(pickle_file)
	return obj
