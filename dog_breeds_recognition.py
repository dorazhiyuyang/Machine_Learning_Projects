#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:01:37 2021

@author: zhiyuyang
"""

import tqdm
      
import numpy as np
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

import matplotlib.pyplot as plt
plt.style.use('default')
from matplotlib import image
from PIL import ImageFile

from keras.utils import np_utils
from keras.preprocessing import image 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 

num_classes = 120
epochs = 200
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, 
                               save_best_only=True)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    raw_targets = np.array(data['target'])
    dog_targets = np_utils.to_categorical(raw_targets, num_classes)
    return dog_files, raw_targets, dog_targets

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


dog_filepaths, dog_raw_targets, dog_targets = load_dataset('Images/')
dogpath_prefix_len = len('Images/n02085620-')
dog_names = [item[dogpath_prefix_len:] for item in sorted(glob("Images/*"))]

X_train, X_test, y_train, y_test = train_test_split(dog_filepaths, dog_targets, test_size=0.2)

half_test_count = int(len(X_test) / 2)
X_valid = X_test[:half_test_count]
y_valid = y_test[:half_test_count]

X_test = X_test[half_test_count:]
y_test = y_test[half_test_count:]

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

train_tensors = paths_to_tensor(X_train).astype(np.float32) / 255
valid_tensors = paths_to_tensor(X_valid).astype(np.float32) / 255
test_tensors = paths_to_tensor(X_test).astype(np.float32) / 255

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', 
                 activation='relu', input_shape=train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
                 
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_tensors, 
          y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=epochs, 
          batch_size=20, 
          callbacks=[checkpointer], 
          verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')


dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = np.sum(np.array(dog_breed_predictions)==np.argmax(y_test, axis=1))/len(dog_breed_predictions)
print('Test Accuracy: {:.4f}'.format(test_accuracy))
