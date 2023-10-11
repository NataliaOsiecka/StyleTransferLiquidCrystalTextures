# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:44:47 2023

@author: Natalia Osiecka
"""
#importing libraries
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

#importing style transfer model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

#importing content image and style image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

content_image = load_image('E:/publikacje/moje/LiquidCrystalsNeuralNetworkAtrs/817-1.jpg')
style_image = load_image('E:/publikacje/moje/LiquidCrystalsNeuralNetworkAtrs/t_0002.png')

#generaiting stylized image
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

cv2.imwrite('generated_imgNematic.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))