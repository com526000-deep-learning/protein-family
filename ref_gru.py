# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense,GRU,RNN,SimpleRNN,Dropout,TimeDistributed,RepeatVector,Bidirectional
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model

def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
    return x_test, y_test

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

x_train=x_train.reshape(-1,28,28)
x_valid=x_valid.reshape(-1,28,28)

# Data Dimension
num_input = 28          # MNIST data input (image shape: 28x28)
timesteps = 28          # Timesteps
n_classes = 10          # Number of classes, one class per digit
 
#Hyperparameters    
learning_rate = 0.001 # The optimization initial learning rate
epochs = 3          # Total number of training epochs
batch_size = 100      # Training batch size
display_freq = 100    # Frequency of displaying the training results
Dropout_rate=0.1
#Network configuration
num_hidden_units = 128  # Number of hidden units of the RNN

model = Sequential()
model.add(GRU(num_hidden_units,return_sequences=True,dropout=0.1, input_shape=(timesteps, num_input),))
model.add(Dropout(rate=Dropout_rate, noise_shape=None, seed=None))
model.add(GRU(num_hidden_units,return_sequences=False,))#, input_shape=(timesteps, num_input)
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
model.summary()

odel.fit(x_train, y_train,
                  batch_size=batch_size, epochs=epochs, shuffle=True)
test_loss = model.evaluate(x_valid, y_valid)