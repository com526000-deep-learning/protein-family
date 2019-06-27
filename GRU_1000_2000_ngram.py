import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import GRU,RNN,SimpleRNN,Dropout,TimeDistributed,RepeatVector,Bidirectional
from keras import optimizers
from keras.callbacks import EarlyStopping

data_1000 = pd.read_csv('data_1000_max2000.csv')
data_1000.head()

x_train, x_test,y_train_1000,y_test_1000 = train_test_split(data_1000['sequence'], data_1000['classification'], test_size = 0.2, random_state = 123)

print(x_train.shape)
print(x_test.shape)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train_1000)
y_test = lb.transform(y_test_1000)
print('number of classes %d'%y_train.shape[1])

def create_ngram_set(input_list, ngram_value=2):
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    # {(4, 9), (4, 1), (1, 4), (9, 4)}
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

ngram_range = 3
maxlen = 512
# maxlen = 1024
batch_size = 512
embedding_dims = 100
epochs = 50

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = len(tokenizer.word_index) + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1
    print(max_features)
    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

top_classes = y_train.shape[1]
dense_size = 256
num_hidden_units = 100  # Number of hidden units of the RNN
Dropout_rate = 0.5
learning_rate = 0.001

# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + len(token_indice) + 1, embedding_dims, input_length = maxlen))

# model 1
# model.add(GRU(num_hidden_units, dropout = Dropout_rate))
# # model.add(Dropout(rate = Dropout_rate))
# model.add(GRU(num_hidden_units))
# model.add(Dropout(rate = Dropout_rate))
# model.add(Dense(top_classes, activation = 'softmax'))

# adam = optimizers.Adam(lr=learning_rate)
# model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

# model.add(GRU(num_hidden_units,return_sequences=True))
model.add(GRU(num_hidden_units))
# model.add(GRU(num_hidden_units,return_sequences=False,))#, input_shape=(timesteps, num_input)
model.add(Dropout(rate=Dropout_rate, noise_shape=None, seed=None))
model.add(Dense(256, activation='relu'))
model.add(Dense(top_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
adam = optimizers.Adam(lr=learning_rate)
history = model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

print(model.summary())

es = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size = batch_size, epochs = epochs, shuffle = True, callbacks=[es])

# test_loss = model.evaluate(x_test, y_test)
# print(test_loss)

acc = np.mean(history.history['acc'])
loss = np.mean(history.history['loss'])
test_acc = np.mean(history.history['val_acc'])
test_loss = np.mean(history.history['val_loss'])

print('train-acc={}, test-acc={}'.format(acc, test_acc))

# model = Sequential()
# model.add(Embedding(len(tokenizer.word_index)+len(token_indice)+1, embedding_dims, input_length=maxlen))

# model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))

# model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(top_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# model.fit(x_train, y_train, validation_data=(c), epochs=10, batch_size=128)

# train_pred = model.predict(x_train)
# test_pred = model.predict(x_test)
# print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
# print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))
# print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))
