# Exercice 2 - Learning Keras and understanding CNN and RNN

# LSTM tutorial : http://colah.github.io/posts/2015-08-Understanding-LSTMs/

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN, RNN
import numpy as np
import pickle

import load_sherlock as sh
import read_write_helpers as rw
import midi_to_data as md
from custom_rnns import MinimalLSTMCell, MinimalRNNCell

# First, the comparisons will be done using a character prediction task on
# a text file : The Adventures of Sherlock Holmes.

[X, y, Xval, yval] = sh.load()

# 1a. Custom RNN layer

print('-------------------- Hand-written RNN ---------------------')
model = Sequential()
model.add(RNN(MinimalRNNCell(256), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=10,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'minimalRNN')

# 1b. Custom LSTM layer

print('-------------------- Hand-written LSTM ---------------------')
model = Sequential()
model.add(RNN(MinimalLSTMCell(256), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=10,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'minimalLSTM')

# 2.  Compare your models to the RNN and LSTM already provided in Keras

# 2a. RNN (default configuration)

print('-------------------- Keras native RNN ---------------------')
model = Sequential()
model.add(SimpleRNN(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=10,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'nativeRNN')

# 2b. LSTM

print('-------------------- Keras native LSTM ---------------------')
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=10,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'nativeLSTM')

#%% 
# 3.  Train all models on your toy datasets and compare performance

[X, y, Xval, yval] = md.load_midi_prediction('toy/dataset/progressions/')

# 3a. Custom RNN layer

print('-------------------- Hand-written RNN ---------------------')
model = Sequential()
model.add(RNN(MinimalRNNCell(256), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=100,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'minimalRNNmidi')

# 3b. Custom LSTM layer

print('-------------------- Hand-written LSTM ---------------------')
model = Sequential()
model.add(RNN(MinimalLSTMCell(256), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=100,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'minimalLSTMmidi')

# 3c. Standard RNN

print('-------------------- Keras native RNN ---------------------')
model = Sequential()
model.add(SimpleRNN(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=100,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'nativeRNNmidi')

# 3d. Standard LSTM

print('-------------------- Keras native LSTM ---------------------')
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
	X,
	y,
	epochs=100,
	batch_size=128,
	validation_data=(
		Xval,
		yval))

rw.save(history.history, 'nativeLSTMmidi')
