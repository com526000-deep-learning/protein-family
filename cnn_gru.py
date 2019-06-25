# pyhton3.5+
# coding: utf-8

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import Dropout, MaxPooling1D, Conv1D
from keras.layers import Embedding, Input
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# import matplotlib.pyplot as plt
# data setup ----------------------------------------------
fname = 'data_1000_max2000.csv'
test_size = 0.2
random_state = 150
max_length = 512

# hyperparameters -----------------------------------------
num_epoch = 20
batch_size = 128
learn_rate = 0.001

conv1_filter = 128
conv1_kernel = 6
pool1_size = 2

conv2_filter = 64
conv2_kernel = 3
pool2_size = 2

GRU_output = 64

drop_prob = 0.1

dense1_size = 256

# load data -----------------------------------------------
data = pd.read_csv(fname)
x_train_1000, x_test_1000, y_train_1000, y_test_1000 = \
train_test_split(data['sequence'], data['classification'], test_size=test_size, random_state=random_state)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train_1000)
y_test = lb.transform(y_test_1000)
print('number of classes %d'%y_train.shape[1])

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(x_train_1000)
# represent input data as word rank number sequences
x_train = tokenizer.texts_to_sequences(x_train_1000)
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = tokenizer.texts_to_sequences(x_test_1000)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

embedding_dim = 16
top_class = y_train.shape[1]


# build model ---------------------------------------------
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=conv1_filter, kernel_size=conv1_kernel, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=pool1_size))
model.add(Conv1D(filters=conv2_filter, kernel_size=conv2_kernel, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=pool2_size))
model.add(GRU(GRU_output))
model.add(Dropout(drop_prob))
model.add(Dense(dense1_size, activation='relu'))
model.add(Dense(top_class, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train ---------------------------------------------------
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
        epochs=num_epoch, batch_size=batch_size)

#model.save('cnn_and_gru.h5')

acc = history.history['acc']
loss = history.history['loss']

test_acc = history.history['val_acc']
test_loss = history.history['val_loss']

for i in range(len(acc)):
    print('train-acc={}'.format(acc[i]), 'test-acc={}'.format(test_acc[i]))


## plot result ---------------------------------------------
#epochs = np.aranga(num_epoch)
#
#plt.plot(epochs, acc, label='train')
#plt.plot(epochs, test_acc, label='test')
#plt.xlabel('epochs')
#plt.ylabel('acc')
#plt.title('accuracy')
#plt.legend()
#plt.savefig('acc.png')
#plt.close()
#
#plt.plot(epochs, loss, label='train')
#plt.plot(epochs, test_loss, label='test')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.title('learning curve')
#plt.legend()
#plt.savefig('loss.png')
#plt.close()


