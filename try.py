from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras
from tensorflow.keras import Model

import matplotlib.pyplot as plt

def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

dataset, info = tfds.load('mnist', data_dir='gs://tfds-data/datasets', with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']
mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
mnist_test = mnist_test.map(convert_types).batch(32)
