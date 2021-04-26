import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
vocab_size = 80000
mx = 470
sample = 12500
reviews_neg = pd.read_csv('data/train_neg.csv')
reviews_pos = pd.read_csv('data/train_pos.csv')
need_neg = reviews_neg['text'][0:sample]
need_pos = reviews_pos['text'][0:sample]
need_all = np.concatenate((need_pos, need_neg), axis=0)
encoding = [one_hot(i, vocab_size) for i in need_all]
padding = pad_sequences(encoding, maxlen=mx, padding='post')
labels = np.zeros(shape=(sample*2, 1))
for i in range(sample):
    labels[i][0] = 0
    labels[i+sample][0] = 1

x_train, x_test, y_train, y_test = train_test_split(padding, labels, test_size=0.2, random_state=42)
print(len(x_train))
print(len(x_train[0]))
print(y_train.shape)
model = Sequential()
embedding_layer = Embedding(vocab_size, 50, input_length=mx)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
print(model.evaluate(x_test, y_test))
