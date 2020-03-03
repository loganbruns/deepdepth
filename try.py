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

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(label, predictions)

@tf.function
def test_step(image, label):
    predictions = model(image)
    t_loss = loss_object(label, predictions)
    test_loss(t_loss)
    test_accuracy(label, predictions)    

EPOCHS = 3
for epoch in range(EPOCHS):
    for image, label in mnist_train:
        train_step(image, label)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 100 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(train_loss.result()))

    for test_image, test_label in mnist_test:
        test_step(test_image, test_label)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))
    model.summary()
    tf.saved_model.save(model, 'tf_model/{}/'.format(epoch))

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
# train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# def create_model():
#     model = tf.keras.models.Sequential([
#         keras.layers.Dense(512, activation='relu', input_shape=(784,)),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
# return model

# # Create a basic model instance
# model = create_model()
# model.summary()
