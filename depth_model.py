
""" Deep Depth Model """

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, LSTM
from tensorflow import keras
from tensorflow.keras import Model

class DepthModel(Model):
    def __init__(self, context_length):
        super(DepthModel, self).__init__()

        self.context_length = context_length

        # Layers
        self.conv1 = Conv2D(12, 5, 2, padding='same', activation='relu')
        self.conv2 = Conv2D(48, 5, 2, padding='same', activation='relu')
        self.conv3 = Conv2D(48, 5, 2, padding='same', activation='relu')
        self.conv4 = Conv2D(48, 5, 2, padding='same', activation='relu')
        self.conv5 = Conv2D(48, 5, 5, padding='same', activation='relu')
        self.layernorm1 = LayerNormalization()
        self.lstm = LSTM(576)
        self.deconv1 = Conv2DTranspose(48, 5, 5, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(48, 5, 2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(48, 5, 2, padding='same', activation='relu')
        self.deconv4 = Conv2DTranspose(48, 5, 2, padding='same', activation='relu')
        self.deconv5 = Conv2DTranspose(12, 5, 2, padding='same', activation='relu')
        self.layernorm2 = LayerNormalization()
        self.conv_final = Conv2D(1, 5, padding='same', activation='linear')

        # Loss
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def call(self, images, focal_lengths, context_length):
        embeddings = []
        for i in range(context_length):
            x = images[:,i,:,:,:]
            focal_length = focal_lengths[:,i]
            focal_length = tf.reshape(focal_length, focal_length.shape + [1, 1, 1])
            focal_length = focal_length * tf.ones(x.shape[:3] + [1])
            
            x = tf.concat([x, focal_length], 3)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.layernorm1(x)
            embeddings.append(x)
        embeddings = tf.stack(embeddings, axis=1)
        embeddings_shape = list(embeddings.shape)
        embeddings = tf.reshape(embeddings, embeddings_shape[:2] + [-1])
        x = self.lstm(embeddings)
        x = tf.reshape(x, [-1] + embeddings_shape[2:])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.layernorm2(x)
        x = tf.squeeze(self.conv_final(x))
        return x

    @tf.function
    def train_step(self, images, depth, focal_lengths, context_length=None):
        if not context_length:
            context_length = self.context_length
        with tf.GradientTape() as tape:
            predictions = self(images, focal_lengths, context_length)
            loss = self.loss_object(depth, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            return predictions

    @tf.function
    def test_step(self, images, depth, focal_lengths, context_length=None):
        if not context_length:
            context_length = self.context_length
        predictions = self(images, focal_lengths, context_length)
        t_loss = self.loss_object(depth, predictions)
        self.test_loss(t_loss)
        return predictions



