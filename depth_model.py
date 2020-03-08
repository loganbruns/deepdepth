
""" Deep Depth Model """

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LayerNormalization
from tensorflow import keras
from tensorflow.keras import Model

class DepthModel(Model):
    def __init__(self):
        super(DepthModel, self).__init__()

        # Layers
        self.conv1 = Conv2D(12, 5, 2, padding='same', activation='relu')
        self.conv2 = Conv2D(48, 5, 2, padding='same', activation='relu')
        self.conv3 = Conv2D(192, 5, 2, padding='same', activation='relu')
        self.conv4 = Conv2D(768, 5, 2, padding='same', activation='relu')
        self.conv5 = Conv2D(3072, 5, 2, padding='same', activation='relu')
        self.layernorm1 = LayerNormalization()
        self.deconv1 = Conv2DTranspose(3072, 5, 2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(768, 5, 2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(192, 5, 2, padding='same', activation='relu')
        self.deconv4 = Conv2DTranspose(48, 5, 2, padding='same', activation='relu')
        self.deconv5 = Conv2DTranspose(12, 5, 2, padding='same', activation='relu')
        self.layernorm2 = LayerNormalization()
        self.conv_final = Conv2D(1, 5, padding='same', activation='linear')

        # Loss
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.layernorm1(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.layernorm2(x)
        x = tf.squeeze(self.conv_final(x))
        return x

    @tf.function
    def train_step(self, image, depth):
        with tf.GradientTape() as tape:
            predictions = self(image)
            loss = self.loss_object(depth, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)

    @tf.function
    def test_step(self, image, depth):
        predictions = self(image)
        t_loss = self.loss_object(depth, predictions)
        self.test_loss(t_loss)
        return predictions



