# pyhton3.5+
# coding: utf-8

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import Dropout, MaxPooling1D, Conv1D
from keras.layers import Embedding, Input
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from save_history import save_history
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

# data setup ----------------------------------------------
# fname = 'data_1000_max2000.csv'
fname = 'data_top10_undersample.csv'
test_size = 0.2
random_state = 150

# hyperparameters -----------------------------------------
num_epoch = 40
batch_size = 256
learning_rate = 0.0005

conv1_filter = 128
conv1_kernel = 6
pool1_size = 2

conv2_filter = 64
conv2_kernel = 3
pool2_size = 2

GRU_output = 64

drop_prob = 0.5

dense1_size = 256

# load data -----------------------------------------------
data = pd.read_csv(fname)
x_train_1000, x_test_1000, y_train_1000, y_test_1000 = \
train_test_split(data['sequence'], data['classification'], test_size=test_size, random_state=random_state)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train_1000)
y_test = lb.transform(y_test_1000)
print('number of classes %d'%y_train.shape[1])

# embedding ------------------------------------
ngram_range = 3
maxlen = 256
embedding_dims = 100

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(x_train_1000)
x_train = tokenizer.texts_to_sequences(x_train_1000)
x_test = tokenizer.texts_to_sequences(x_test_1000)
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    start_index = len(tokenizer.word_index) + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1
    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

top_class = y_train.shape[1]
# embedding_dim = 16


# build model ---------------------------------------------
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + len(token_indice) + 1, embedding_dims, input_length = maxlen))
model.add(Conv1D(filters=conv1_filter, kernel_size=conv1_kernel, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=pool1_size))
model.add(Conv1D(filters=conv2_filter, kernel_size=conv2_kernel, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=pool2_size))

# model.add(Dense(512, activation='relu'))

model.add(GRU(GRU_output))
model.add(Dropout(drop_prob))
model.add(Dense(dense1_size, activation='relu'))
model.add(Dense(top_class, activation='sigmoid'))
adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
model.summary()

# train ---------------------------------------------------
es = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
        epochs=num_epoch, batch_size=batch_size, callbacks=[es])
#model.save('cnn_and_gru.h5')
save_history((history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_acc']),\
 'cnn_gru_ngram_10')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))
print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))

# Compute confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))

# Plot normalized confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(lb.classes_))
plt.xticks(tick_marks, lb.classes_, rotation=90)
plt.yticks(tick_marks, lb.classes_)
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
