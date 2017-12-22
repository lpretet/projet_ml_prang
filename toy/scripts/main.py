"""Main file.

Automatically builds the toy dataset.

Example
-------
How to use this code

	$ python toy/scripts/main.py

"""

import progressions_list as pr
import progressions_to_midi as gen

prog_list = pr.progressions_list()
gen.generate_all_midi_progressions(prog_list, 'toy/dataset/progressions/')