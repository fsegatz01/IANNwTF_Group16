# -*- coding: utf-8 -*-
"""data_processing_library

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cBzF4oA5yftkohJD7sKkp__BNtgSidD4
"""

import h5py
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt
from google.colab import drive
import os

"""##PREPROCESSING

ECO
"""

############# PREPROCESS ECO #################

def eco_data_pip(data, dataset_type, batch_size):

  #create one-hot targets
  data = data.map(lambda img, target: (img, tf.one_hot(target, depth=100)))

  #cifar10 = tf.data.Dataset.from_tensor_slices(cifar10)
  data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
  #data = data.map(lambda img, target: ((img/128.)-1., target))
  data = data.map(lambda img, target: ((img/255), target))

  resize_layer = tf.keras.layers.Resizing(32, 32)

  # Apply the Resizing layer to the datasets
  data = data.map(lambda img, target: (resize_layer(img), target))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all

  data = data.cache()
  # shuffle, batch, prefetch
  data = data.shuffle(1000)
  #data = data.map(lambda img, target: (tf.expand_dims(img, 0), target))
  data = data.batch(32)
  data = data.prefetch(20)
  return data

"""NSD"""

#data augmentation
#add slightly transformed/rotated copies of already included data

def data_augmentation(data):

  augmentation_model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
  ])

  augmentation_model(data)

############# PREPROCESS  NSD ################

def preprocess_NSD(x_train_data):

    # convert the 2d array output labels into 1D array
    x_train_data = x_train_data.astype('float32')

    # normalizing the training and testing data
    x_train_data /= 255.0
    x_train_data = resize_layer(x_train_data)
    x_train_data = augmentation_model(x_train_data)
    
    return x_train_data

"""CIFAR"""

###### PREPROCESS CIFAR #######

def preprocess_cifar(data, batch_size, resize, data_augmentation):

  if resize:
    # Apply the Resizing layer to the datasets
    data = data.map(lambda img, target: (resize_layer(img), target))

  if data_augmentation: 
    #data augmentation
    data = data.map(lambda img, target: (augmentation_model(img), target))

  #casting image value to float32
  data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, bringing image values from range [0, 255] to [-1, 1]
  data = data.map(lambda img, target: ((img/128.)-1., target))
  #creating one hot vectors for the labels
  data = data.map(lambda img, target: (img, tf.one_hot(target, depth=100)))
  #cache this progress in memory
  #shuffle, batch, prefetch
  data = data.cache().shuffle(1000).batch(batch_size).prefetch(20)

  return data

"""##VISUALIZATION

ECO & CIFAR
"""

########## VISUALIZE ECO AND CIFAR #######################
def visualize(images, labels):
    fig, axes = plt.subplots(3,2, figsize=(10,10))
    fig.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i*3000])
        ax.set_title(labels)
        ax.axis("off")

"""NSD"""

### VISUALIZE ###

def visualize_NSD(images):
    fig, axes = plt.subplots(3,2, figsize=(8,8))
    fig.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].reshape(32, 32, 3))
        ax.set_title('Training image: ' + str(i+1))
        ax.axis("off")