"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/7 18:43
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import time
import cv2
import paddlehub as hub
start_time = time.time()
model = hub.Module(name='animegan_v2_shinkai_33', use_gpu=True)
result = model.style_transfer(images=[cv2.imread('image/test.jpg')], visualization=True)
end_time = time.time()
sum_time = end_time - start_time
print("Inference in ", int(sum_time), 'seconds')

