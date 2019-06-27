#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

# In[3]:


data_1000 = pd.read_csv('data_top10_undersample.csv')
# data_1000 = pd.read_csv('data_1000_max2000.csv')
# data_1000 = pd.read_csv('data_1000_max1200.csv')


# In[4]:


data_1000.head()


# In[5]:


X_train_1000, X_test_1000,y_train_1000,y_test_1000 = train_test_split(data_1000['sequence'], data_1000['classification'], test_size = 0.2, random_state = 123)


# In[6]:


print(X_train_1000.shape)
print(X_test_1000.shape)


# In[7]:


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train_1000)
y_test = lb.transform(y_test_1000)
print('number of classes %d'%y_train.shape[1])


# In[8]:


# maximum length of sequence, everything afterwards is discarded
max_length = 500

#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_1000)
#represent input data as word rank number sequences
X_train = tokenizer.texts_to_sequences(X_train_1000)
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = tokenizer.texts_to_sequences(X_test_1000)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)


# In[9]:


embedding_dim = 16
top_classes = y_train.shape[1]
# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=256, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(top_classes, activation='softmax'))
adam = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
print(model.summary())


# In[10]:


es = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
# epochs=15, # batch_size=128
history = model.fit(X_train, y_train,  batch_size=128, verbose=1, validation_split=0.15, callbacks=[es], epochs=25)

# In[11]:


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))
print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))


# In[ ]:




