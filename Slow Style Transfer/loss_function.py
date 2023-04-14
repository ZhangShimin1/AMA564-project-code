"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/2 17:02
"""
from tensorflow.keras import backend as K
from preprocessing import img_height, img_width


def content_loss(base, combination):
    return K.sum(K.square(combination-base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    print(x[:, :img_height-1, :img_width-1, :], x[:, 1:, :img_width-1, :])
    a = K.square(
        x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :]
    )
    b = K.square(
        x[:, img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :]
    )
    return K.sum(K.pow(a+b, 1.25))