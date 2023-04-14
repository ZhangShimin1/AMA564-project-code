"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/18 13:31
"""
import tensorflow as tf
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from models import network


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

inference_img_path = "./img/frame/street.jpg"
weights_path = "./checkpoints/candy/1000"

width, height = tf.keras.preprocessing.image.load_img(inference_img_path).size
target_height = 512
target_width = int(math.ceil(width / height) * target_height)


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_height, target_width), interpolation="bicubic")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)


def deprocess_image(x):
    x = x.numpy().reshape((target_height, target_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    #  'BGR'->'RGB'
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')


feedforward_network = network(scale=16)
feedforward_network.load_weights(weights_path)
img_tensor = preprocess_image(inference_img_path)

feedforward_network(img_tensor)
start_time = time.time()
result_map = deprocess_image(feedforward_network(img_tensor))
print(result_map)
end_time = time.time()
print("stylish time in %d ms" % (int(end_time - start_time)))


plt.imshow(result_map)
plt.axis('off')
plt.margins(0, 0)
plt.savefig("./img/result/res_dog.jpg", bbox_inches='tight')

plt.show()




