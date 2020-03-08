from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from tensorflow import keras
from tensorflow.keras import Model

import numpy as np

from nyu import NYUv2Dataset

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(12, 5, 2, padding='same', activation='relu')
        self.conv2 = Conv2D(48, 5, 2, padding='same', activation='relu')
        self.conv3 = Conv2D(192, 5, 2, padding='same', activation='relu')
        self.conv4 = Conv2D(768, 5, 2, padding='same', activation='relu')
        self.conv5 = Conv2D(3072, 5, 2, padding='same', activation='relu')
        self.deconv1 = Conv2DTranspose(3072, 5, 2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(768, 5, 2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(192, 5, 2, padding='same', activation='relu')
        self.deconv4 = Conv2DTranspose(48, 5, 2, padding='same', activation='relu')
        self.deconv5 = Conv2DTranspose(12, 5, 2, padding='same', activation='relu')
        self.conv_final = Conv2D(1, 5, padding='same', activation='relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = tf.squeeze(self.conv_final(x))
        return x

model = MyModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(image, depth):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(depth, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

@tf.function
def test_step(image, depth):
    predictions = model(image)
    t_loss = loss_object(depth, predictions)
    test_loss(t_loss)
    return predictions

def main():
    """start main training loop"""

    # Set up experiment dir
    experiment_name = 'baseline'
    experiment_dir = f'./experiments/{experiment_name}'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Set up check pointing
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, f'{experiment_dir}/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Set up experimental logging
    train_log_dir = f'{experiment_dir}/logs/train'
    test_log_dir = f'{experiment_dir}/logs/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)        

    # Load NYUv2 depth dataset
    train, val, test = NYUv2Dataset('data/nyu_depth_v2.tfrecord')

    # Start training loop
    EPOCHS = 1000
    once_per_train = False
    for epoch in range(EPOCHS):
        starttime = time.time()
        startstep = int(ckpt.step)
        for image, label in train:
            train_step(image, label)
            ckpt.step.assign_add(1)

            if not once_per_train:
                model.summary()
                once_per_train = True

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=int(ckpt.step))
                
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("Training loss {:1.2f}".format(train_loss.result()))

        with test_summary_writer.as_default():
            for test_image, test_label in val:
                test_predictions = test_step(test_image, test_label)
                tf.summary.image('context_images', test_image, step=int(ckpt.step))
                # tf.summary.image('real_depth_map', test_label, step=int(ckpt.step))
                # tf.summary.image('pred_depth_map', test_predictions, step=int(ckpt.step))

            print(f"{int(ckpt.step)}: test loss={test_loss.result()}")
            tf.summary.scalar('loss', test_loss.result(), step=int(ckpt.step))

        template = 'Epoch {}, Loss: {}, Test Loss: {}, Sec/Iters: {}'
        print (template.format(epoch+1,
                               train_loss.result(),
                               test_loss.result(),
                               (time.time()-starttime)/(int(ckpt.step)-startstep)))
        tf.saved_model.save(model, f'{experiment_dir}/tf_model/{int(ckpt.step)}/')

if __name__ == '__main__':
    main()
