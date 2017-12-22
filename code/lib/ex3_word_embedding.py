### Exercice 3 - Word embedding for music : Understanding the reference paper

## 1. Re-implement the tutorial given

# The goal is to use a CNN for sequence classification in the IMDB dataset.
# Movies are represented by a sequence of words (the review).
# It is a binary classification problem : movie -> positive or negative review.

import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Flatten, Reshape
from keras.layers import LSTM, Convolution1D, Dropout, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import read_write_helpers as rw
import midi_to_data as md
import training_helpers as tr

# Using keras to load the dataset with the top_words
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Pad the sequence to the same length
max_review_length = 1600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using native word embedding from Keras
embedding_vector_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
model.save('code/models/ImdbCNNModel1.h5')
rw.save(history.history, 'ImdbCNN1')

#%% 
## 2. 3. Extend and apply the model to musical data (your toy dataset) and compare performances

(X_train, y_train, X_test, y_test) = md.load_midi_prediction('toy/dataset/progressions/')

model = Sequential()

# Convolutional model (3x conv, flatten, 1x dense, 1xLSTM)
# Input : 3D tensor with shape: (batch_size, steps, input_dim)
model.add(Convolution1D(64, 3, padding='same', input_shape= (X_train.shape[1], X_train.shape[2])))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(LSTM(y_train.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tr.frame_loss, 'accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
model.save('code/models/MidiCNNModel.h5')
rw.save(history.history, 'MidiCNN')

#%% 
## 4.  Analyze and explain the behavior of the models for different properties/architectures

# Since the previous model showed very bad results when applied directly to musical data, 
# we made some modifications.
# Here we changed the optimized, the filters size, the activation function, and we added a Max Pooling layer.

model = Sequential()
model.add(Convolution1D(80, 16, padding='same', input_shape= (X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(5))
model.add(Convolution1D(50, 16, padding='same'))
model.add(Convolution1D(30, 16, padding='same'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='relu'))
model.add(LSTM(y_train.shape[1]))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[tr.frame_loss, 'accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
model.save('code/models/MidiCNNModel2.h5')
rw.save(history.history, 'MidiCNN2')

#%% 
# Then we used a TimeDistributed architecture to better fit the temporal organisation of our data.

latent_space_dim = 10

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

model = Sequential()
model.add(TimeDistributed(Convolution1D(80, 16, padding='same'), input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(TimeDistributed(Convolution1D(50, 16, padding='same')))
model.add(TimeDistributed(Convolution1D(30, 16, padding='same')))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(latent_space_dim)))
model.add(Reshape((10,1,latent_space_dim)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(y_train.shape[1]))

model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=[tr.frame_loss, 'accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

model.save('code/models/MidiCNNModel3.h5')
rw.save(history.history, 'MidiCNN3')


