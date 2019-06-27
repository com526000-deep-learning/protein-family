import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AtrousConvolution1D, SpatialDropout1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
from keras.callbacks import EarlyStopping
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools

# Merge the two Data set together
# Drop duplicates (unlike the original script)
# df = pd.read_csv('pdb_data_no_dups.csv').merge(pd.read_csv('pdb_data_seq.csv'), how='inner', on='structureId').drop_duplicates(["sequence"]) # ,"classification"
df = pd.read_csv('pdb_data_no_dups.csv').merge(pd.read_csv('pdb_data_seq.csv'), how='inner', on='structureId')
# Drop rows with missing labels
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]
# select proteins
df = df[df.macromoleculeType_x == 'Protein']
df.reset_index()
print(df.shape)
# print(df.head())


# df = df.loc[df.residueCount_x>1200]
# print(df.shape[0])


# count numbers of instances per class
cnt = Counter(df.classification)
# select only K most common classes! - was 10 by default
top_classes = 10
# sort classes
sorted_classes = cnt.most_common()[:top_classes]
classes = [c[0] for c in sorted_classes]
counts = [c[1] for c in sorted_classes]
# print("at least " + str(counts[-1]) + " instances per class")

# apply to dataframe
# print(str(df.shape[0]) + " instances before")
df = df[[c in classes for c in df.classification]]
# print(str(df.shape[0]) + " instances after")

seqs = df.sequence.values
lengths = [len(s) for s in seqs]

# visualize
# fig, axarr = plt.subplots(1,2, figsize=(20,5))
# axarr[0].bar(range(len(classes)), counts)
# plt.sca(axarr[0])
# plt.xticks(range(len(classes)), classes, rotation='vertical')
# axarr[0].set_ylabel('frequency')

# axarr[1].hist(lengths, bins=50, density=False)
# axarr[1].set_xlabel('sequence length')
# axarr[1].set_ylabel('# sequences')
# plt.show()


from sklearn.preprocessing import LabelBinarizer

# Transform labels to one-hot
lb = LabelBinarizer()
Y = lb.fit_transform(df.classification)


# maximum length of sequence, everything afterwards is discarded! Default 256
max_length = 256

#create and fit tokenizer
# tokenizer = Tokenizer(char_level=True)
# tokenizer.fit_on_texts(seqs)
# #represent input data as word rank number sequences
# X = tokenizer.texts_to_sequences(seqs)
# X = sequence.pad_sequences(X, maxlen=max_length)
X_train, X_test, y_train, y_test = train_test_split(seqs, Y, test_size=.2)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))
vect.fit(X_train)
X_train = vect.transform(X_train).toarray()
X_test = vect.transform(X_test).toarray()


embedding_dim = 16 # orig 8

# create the model
model = Sequential()
# model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))

model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', dilation_rate=1, input_shape=X_train.shape))
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')) 
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))


model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))) # 128
# model.add(BatchNormalization())

# model.add(Dropout(0.5))

model.add(Dense(top_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


es = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
# epochs=15, # batch_size=128
history = model.fit(X_train, y_train,  batch_size=128, verbose=1, validation_split=0.15, callbacks=[es], epochs=25)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))

# print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # Compute confusion matrix
# cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))

# # Plot normalized confusion matrix
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# np.set_printoptions(precision=2)
# plt.figure(figsize=(10,10))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# tick_marks = np.arange(len(lb.classes_))
# plt.xticks(tick_marks, lb.classes_, rotation=90)
# plt.yticks(tick_marks, lb.classes_)
# #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# #    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))

