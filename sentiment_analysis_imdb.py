#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:09:26 2022

@author: zhiyuyang
"""

import numpy 
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)

top_words = 5000

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=top_words)
review_max_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=review_max_length)
X_test = sequence.pad_sequences(X_test, maxlen=review_max_length)

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vecor_length, input_length=review_max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}".format((scores[1]*100)))