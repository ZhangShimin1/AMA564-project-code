"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/2 17:00
"""
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19

import numpy as np
import cv2

target_img_path = 'img/musk.jpg'
style_reference_img_path = 'img/bjs.jpg'

width, height = load_img(target_img_path).size
img_height = 400
img_width = int(width * img_height / height)


def get_RGB(img_path):
    image_getRGB = cv2.imread(img_path)
    image_getRGB = cv2.cvtColor(image_getRGB, cv2.COLOR_BGR2RGB)
    value_B = image_getRGB[:, :, 0].mean()
    value_G = image_getRGB[:, :, 1].mean()
    value_R = image_getRGB[:, :, 2].mean()
    return value_R, value_G, value_B


def preprocess_img(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_img(img, img_path):
    value_R, value_G, value_B = get_RGB(img_path)
    img[:, :, 0] += value_B
    img[:, :, 1] += value_G
    img[:, :, 2] += value_R
    img = img[:, :, ::-1]  # BGR-->RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img


# print(value_R, value_G, value_B)