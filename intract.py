import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
# parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--input', default='examples/temp.jpg', type=str, help='Input filename or folder.')
# args = parser.parse_args()

image_path = 'examples/office2.jpg'

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format('nyu.h5'))

# Input images
inputs = load_images( glob.glob(image_path) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
print(outputs.shape)

outputs = outputs*10

img = cv2.imread(image_path)
img_copy = img.copy()
print(img.shape)
cv2.imshow('image', img_copy)


def mouse_click(event, x, y, 
                flags, param):
    print(x, y)
    img_copy = img.copy()
    # font for right click event
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    dis = round(outputs[0][int(y/2)][int(x/2)][0], 2)
    RB = str(dis)
        
    # display that right button 
    # was clicked.
    cv2.putText(img_copy, RB, (x, y),
                font, 1, 
                (0, 255, 255),
                2)
    cv2.imshow('image', img_copy)

cv2.setMouseCallback('image', mouse_click)

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
