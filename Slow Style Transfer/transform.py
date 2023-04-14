"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/2 17:03
"""
import cv2
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b
import time

from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg19

from preprocessing import preprocess_img, \
    img_height, img_width, deprocess_img
from loss_function import content_loss, style_loss, total_variation_loss

tf.compat.v1.disable_eager_execution()
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
transform_img = K.placeholder((1, img_height, img_width, 3))


def model_prepare(target_path, style_path):
    target_img = K.constant(preprocess_img(target_path))
    style_reference_img = K.constant(preprocess_img(style_path))
    input_tensor = K.concatenate([target_img, style_reference_img, transform_img], axis=0)
    print("input tensor", input_tensor)
    model_vgg19 = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    return model_vgg19


def compute_initial_loss_grads(target_path, style_path):
    model = model_prepare(target_path, style_path)
    output_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    loss = K.variable(0.)  # 最终损失值
    print("initial loss: ", loss)
    layer_features = output_dict[content_layer]
    print("content feature：", layer_features)
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(target_image_features, combination_features)  # 加内容损失
    print("content loss: ", loss)
    for layer_name in style_layers:  # 加风格损失
        layer_features = output_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    print("content loss + style loss: ", loss)
    # 加变异损失，得到最终损失函数值
    loss += total_variation_weight * total_variation_loss(transform_img)
    print("content loss + style loss + variation loss", loss)
    outputs = [loss]
    print("transform img tensor: ", transform_img)

    grads = tf.gradients(loss, transform_img)
    # with tf.GradientTape() as gtape:
    #     gtape.watch(transform_img)
    #     grads = gtape.gradient(loss, transform_img)

    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
    print(outputs)
    return outputs


target_img_path = 'img/street.jpg'
style_reference_img_path = 'img/starry_night.jpg'
initial_loss_grads = compute_initial_loss_grads(target_img_path, style_reference_img_path)
fetch_loss_and_grads = K.function([transform_img], initial_loss_grads)


# 类上方代码不动
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        # fetch_loss_and_grads = K.function([transform_img], initial_loss_grads)
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = np.array(outs[1]).flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


result_prefix = './transform/result'
iterations = 100

x = preprocess_img(target_img_path)  # 目标图片路径
x = x.flatten()  # 展开，应用l-bfgs
min_value = 9999999999

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    # 在生成图片上运行L-BFGS优化；注意传递计算损失和梯度值必须为两个不同函数作为参数
    x, current_loss, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_img(img, target_img_path)
    fname = result_prefix + str(i) + '.jpg'
    if current_loss <= min_value:
        min_value = current_loss
        print('Current minimum loss value:', min_value)
        cv2.imwrite(fname, img)
        print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

