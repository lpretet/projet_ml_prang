"""Main file.

Launches all scripts necessary to understanding our work, starting at exercise 2.
Please refer to toy/scripts/main for exercice 1 (toy dataset generation).

Example
-------
How to use this code : first build the dataset, then execute the code.

	$ python toy/scripts/main.py
	$ python code/lib/main.py

"""

import numpy as np
import read_write_helpers as rw
import matplotlib.pyplot as plt
import training_helpers as tr
import visualize_embedding as vis
import keras.models

# ~~~~~~~~~~~~  Exercise 2 : understanding Keras, RNNs and LSTMs

# Here we will load pre-trained models, visualize their training history, and compare them with each other.
# The goal is to try to understand their behaviour.
# The code used to generate the training history data lies in ex2_keras
# (but can be very long to run).

#%% ~~~~~ In this first section, one can compare the several models trained for
# text prediction.

mLSTM = rw.load_pickled('code/models/minimalLSTM.pickle')
nLSTM = rw.load_pickled('code/models/nativeLSTM.pickle')
mRNN = rw.load_pickled('code/models/minimalRNN.pickle')
nRNN = rw.load_pickled('code/models/nativeRNN.pickle')

f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)

a1.plot([mLSTM['acc']])
a1.set_title('Custom LSTM')
a1.plot(mLSTM['val_acc'])
a1.legend(['acc', 'val_acc'])

a2.plot(nLSTM['acc'])
a2.set_title('Native LSTM')
a2.plot(nLSTM['val_acc'])
a2.legend(['acc', 'val_acc'])

a3.plot(mRNN['acc'])
a3.set_title('Custom RNN')
a3.plot(mRNN['val_acc'])
a3.legend(['acc', 'val_acc'])

a4.plot(nRNN['acc'])
a4.set_title('Native RNN')
a4.plot(nRNN['val_acc'])
a4.legend(['acc', 'val_acc'])

plt.show()

#%%  ~~~~~ In this second section, one can compare the several models trained for
# midi chords prediction.

mLSTMmidi = rw.load_pickled('code/models/minimalLSTMmidi.pickle')
nLSTMmidi = rw.load_pickled('code/models/nativeLSTMmidi.pickle')
mRNNmidi = rw.load_pickled('code/models/minimalRNNmidi.pickle')
nRNNmidi = rw.load_pickled('code/models/nativeRNNmidi.pickle')

f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)

a1.plot(mLSTMmidi['acc'][:10])
a1.set_title('Custom LSTM')
a1.plot(mLSTMmidi['val_acc'][:10])
a1.legend(['acc', 'val_acc'][:10])

a2.plot(nLSTMmidi['acc'][:10])
a2.set_title('Native LSTM')
a2.plot(nLSTMmidi['val_acc'][:10])
a2.legend(['acc', 'val_acc'][:10])

a3.plot(mRNNmidi['acc'][:10])
a3.set_title('Custom RNN')
a3.plot(mRNNmidi['val_acc'][:10])
a3.legend(['acc', 'val_acc'][:10])

a4.plot(nRNNmidi['acc'][:10])
a4.set_title('Native RNN')
a4.plot(nRNNmidi['val_acc'][:10])
a4.legend(['acc', 'val_acc'][:10])

plt.show()

# ~~~~~~~~~~~~ Exercise 3 : Word embeddings for music

# The code used to generate the training history data lies in ex3_word_embedding
# (but can be very long to run).

#%% ~~~~~ In this first section, we will show the behavior of a Convolutional Network trained 
# for binary sentiment classification.

# Here we visualize the training history of the CNN as presented in the tutorial.

ImdbCNN = rw.load_pickled('code/models/ImdbCNN1.pickle')
plt.plot(ImdbCNN['acc'])
plt.title('CNN on IMDB')
plt.plot(ImdbCNN['val_acc'])
plt.legend(['acc', 'val_acc'])

plt.show()

#%% ~~~~~ In this second section, we will show the behavior the same Convolutional Network
# trained for midi chords prediction.

MidiCNNbase = rw.load_pickled('code/models/MidiCNN.pickle')
MidiCNNok = rw.load_pickled('code/models/MidiCNNmain.pickle')

f, (a1, a2) = plt.subplots(1, 2)

a1.plot(MidiCNNbase['acc'])
a1.set_title('CNN on Midi toy dataset')
a1.plot(MidiCNNbase['val_acc'])
a1.legend(['acc', 'val_acc'])

# Because of the bad results obtained with the original settings, we modified the architecture of the network.
# Now the results are : 

a2.plot(MidiCNNok['acc'])
a2.set_title('CNN on Midi toy dataset with TimeDistributed layers')
a2.plot(MidiCNNok['val_acc'])
a2.legend(['acc', 'val_acc'])

plt.show()

#%% ~~~~~ Now we will visualize some basic properties of our embedding space.
# Here we plot a 2D visualisation our dataset using T-SNE dimensionality reduction algorithm.
# We colored in red the four chords from the 'Eiffel' progression.
# This example is a bit longer than the other ones, as it requires to load the data and perform a TSNE algorithm.

(X, labels, sizes) = vis.load_midi_simple_visualisation('toy/dataset/progressions/', ['Eiffel', 't0'])
model = keras.models.load_model('code/models/MidiCNNModel3.h5', custom_objects={'frame_loss': tr.frame_loss})
# This model (featuring TimeDistributed layers) needs some preliminary reshaping.
X = np.reshape(X, (X.shape[0], X.shape[1], 1, X.shape[2]))
vis.visualize_model_pattern(X, labels, sizes, model)








