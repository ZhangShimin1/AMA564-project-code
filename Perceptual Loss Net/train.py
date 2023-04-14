"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/18 13:30
"""
from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfd
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt

from models import network


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

style_reference_path = "./styles/candy.jpg"

content_loss_weight = 1e0
style_loss_weight = 2e-5
total_variation_loss_weight = 2e-4

width = 256
height = 256


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(width, height), interpolation="bicubic")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


def deprocess_image(x):
    x = x.numpy().reshape((width, height, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    #  'BGR'->'RGB'
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


@tf.function
def gram_matrix(x):
    """ Computes the gram matrix of an image tensor (feature-wise outer product)."""
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations


@tf.function
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return style_loss_weight * tf.reduce_mean(tf.square(S - C))


@tf.function
def content_loss(base, combination):
    return content_loss_weight * tf.reduce_mean(tf.square(combination - base))


@tf.function
def total_variation_loss(x):
    return total_variation_loss_weight * tf.image.total_variation(x)


content_layers = ['block5_conv2']  # deeper layers for the content features
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']  # lower layers for the style loss


class LossEval(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, style_reference):
        super(LossEval, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_reference = style_reference
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, base_image, combined_image):
        base_image_features = self.vgg(base_image)
        style_reference_features = self.vgg(self.style_reference)
        combination_features = self.vgg(combined_image)

        loss = content_loss(base_image_features[-1], combination_features[-1])
        loss += total_variation_loss(combined_image)
        for i in range(self.num_style_layers):
            sl = style_loss(style_reference_features[i], combination_features[i])
            loss += (sl / self.num_style_layers)
        return loss


loss_eval = LossEval(style_layers, content_layers, tf.constant(preprocess_image(style_reference_path)))


def call_loss_model(y_true, y_pred):
    return loss_eval(y_true, y_pred)


feedforward_network = network(scale=16)
feedforward_network.compile(loss=call_loss_model, optimizer="adam")


dataset, info = tfd.load(name="caltech101", split="train", with_info=True)


def resize(x):
    return tf.image.resize(x, (width, height))


def get_image(x):
    return x["image"]


batch_size = 4
sample_num = info.splits["train"].num_examples
data = dataset.repeat()
data = data.map(get_image).map(resize)
data = data.map(tf.keras.applications.vgg19.preprocess_input)
data = data.shuffle(3059).batch(batch_size)

itr = 0
fig, ax = plt.subplots(3, 2)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
for batch in data:
    # Note: not training an identity function!
    # Custom loss function for the output is wrt. the input image
    feedforward_network.fit(batch, batch)
    fig.suptitle("Iteration %d" % itr)
    if itr % 10 == 0:
        for j in range(3):
            plt.subplot(321+2*j)
            plt.cla()
            plt.imshow(deprocess_image(batch[j]))
            plt.axis('off')
            plt.subplot(322+2*j)
            plt.cla()
            plt.imshow(deprocess_image(feedforward_network(tf.expand_dims(batch[j], axis=0))))
            plt.axis('off')
        plt.pause(0.01)

    if itr % 100 == 0:
        feedforward_network.save_weights("checkpoints/%d" % itr)
        plt.savefig("checkpoints/%d.png" % itr)
    # train for about 2 epochs
    if itr * batch_size > 2 * sample_num:
        break
    itr += 1
    print(itr)



