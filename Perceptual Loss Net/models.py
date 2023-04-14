"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/18 13:30
"""
import tensorflow as tf
import tensorflow_addons as tfa


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.layers = [
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3)),
            tfa.layers.InstanceNormalization(),
            tf.keras.activations.relu,
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3)),
            tfa.layers.InstanceNormalization()
        ]
        for (i, layer) in enumerate(self.layers):
            self.__setattr__("layer_%d" % i, layer)

    def call(self, x):
        # Residual: add input to output
        res = tf.keras.layers.Cropping2D(((2, 2), (2, 2)))(x)
        for layer in self.layers:
            x = layer(x)
        return tf.keras.backend.sum([x, res], axis=0)


def network(scale=32):
    input_x = tf.keras.layers.Input((None, None, 3))
    padding_x = tf.pad(input_x, [[0, 0], [40, 40], [40, 40], [0, 0]], 'REFLECT')
    # 3 convolutional layers
    C1 = tf.keras.layers.Conv2D(filters=scale, kernel_size=(9, 9), strides=1, padding='same')(padding_x)
    C1 = tfa.layers.InstanceNormalization()(C1)
    C1 = tf.keras.activations.relu(C1)
    C2 = tf.keras.layers.Conv2D(filters=scale*2, kernel_size=(3, 3), strides=2, padding='same')(C1)
    C2 = tfa.layers.InstanceNormalization()(C2)
    C2 = tf.keras.activations.relu(C2)
    C3 = tf.keras.layers.Conv2D(filters=scale*4, kernel_size=(3, 3), strides=2, padding='same')(C2)
    C3 = tfa.layers.InstanceNormalization()(C3)
    C3 = tf.keras.activations.relu(C3)
    #  5 residual blocks
    r1 = ResidualBlock(scale*4)(C3)
    r2 = ResidualBlock(scale * 4)(r1)
    r3 = ResidualBlock(scale * 4)(r2)
    r4 = ResidualBlock(scale * 4)(r3)
    r5 = ResidualBlock(scale * 4)(r4)
    #  3 up sampling layers
    S1 = tf.keras.layers.Conv2DTranspose(filters=scale * 2, kernel_size=(3, 3), strides=2, padding="same")(r5)
    S1 = tfa.layers.InstanceNormalization()(S1)
    S1 = tf.keras.activations.relu(S1)
    S2 = tf.keras.layers.Conv2DTranspose(filters=scale, kernel_size=(3, 3), strides=2, padding="same")(S1)
    S2 = tfa.layers.InstanceNormalization()(S2)
    S2 = tf.keras.activations.relu(S2)
    S3 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(9, 9), strides=1, padding="same")(S2)
    S3 = tfa.layers.InstanceNormalization()(S3)
    S3 = tf.keras.activations.tanh(S3)
    output_y = S3 * 127.5

    model = tf.keras.Model(inputs=input_x, outputs=output_y)
    model.summary()
    return model

