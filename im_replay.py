import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.models import load_model
import keras
import sys
import time
import numpy as np

import airsim

import keras.backend as K
from keras.preprocessing import image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


im_model = keras.models.load_model('./im_model/model_rgb.h5')
im_model._make_predict_function()
graph = tf.get_default_graph()

# Connect to AirSim 
client = airsim.CarClient(ip="127.0.0.2")
client.confirmConnection()

# Initialize image buffer
image_buf = np.zeros((1, 61, 184, 3))

def get_image():
    """
    Get image from AirSim client
    """
    while True:
        try:
            im_image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            im_image1d = np.fromstring(im_image_response.image_data_uint8, dtype=np.uint8)
            im_image_rgb = im_image1d.reshape(im_image_response.height, im_image_response.width, 3)
            break
        except:
            print ("Image Error in IM")
    return im_image_rgb[40:101,36:220,0:3].astype(float)

#def get_im_response():    
def get_im_response_backup():
    # Update throttle value according to steering angle

    image_buf[0] = get_image()
    image_buf[0] /= 255 # Normalization
    
    # Prediction
    im_model_output = im_model.predict([image_buf])

    output = im_model_output[0][0]

    if (im_model_output[0][0] < 0.1):
        im_action = 22

    elif (im_model_output[0][0] < 0.05):
        im_action = 21

    elif (im_model_output[0][0] < 0.1):
        im_action = 20

    elif (im_model_output[0][0] < 0.15):
        im_action = 19

    elif (im_model_output[0][0] < 0.2):
        im_action = 18

    elif (im_model_output[0][0] < 0.25):
        im_action = 17

    elif (im_model_output[0][0] < 0.3):
        im_action = 16

    elif (im_model_output[0][0] < 0.35):
        im_action = 15

    elif (im_model_output[0][0] < 0.4):
        im_action = 14

    elif (im_model_output[0][0] < 0.45):
        im_action = 13

    elif (im_model_output[0][0] < 0.5):
        im_action = 12
    
    elif (im_model_output[0][0] < 0.55):
        im_action = 11

    elif (im_model_output[0][0] < 0.6):
        im_action = 10

    elif (im_model_output[0][0] < 0.65):
        im_action = 9

    elif (im_model_output[0][0] < 0.7):
        im_action = 8

    elif (im_model_output[0][0] < 0.75):
        im_action = 7
    
    elif (im_model_output[0][0] < 0.8):
        im_action = 6

    elif (im_model_output[0][0] < 0.85):
        im_action = 5

    elif (im_model_output[0][0] < 0.9):
        im_action = 4

    elif (im_model_output[0][0] < 0.95):
        im_action = 3

    else:
        im_action = 12

    return im_action

def get_im_response():
    # Update throttle value according to steering angle

    image_buf[0] = get_image()
    image_buf[0] /= 255 # Normalization
    
    # Prediction
    im_model_output = im_model.predict([image_buf])

    output = im_model_output[0][0]

    if (im_model_output[0][0] < 0.125):
        im_action = 5

    elif (im_model_output[0][0] < 0.250):
        im_action = 4

    elif (im_model_output[0][0] < 0.375):
        im_action = 3

    elif (im_model_output[0][0] < 0.5):
        im_action = 6

    elif (im_model_output[0][0] < 0.625):
        im_action = 2

    elif (im_model_output[0][0] < 0.750):
        im_action = 1

    else:
        im_action = 0

    
    return im_action

def get_im_response_throttle():
    # Update throttle value according to steering angle

    image_buf[0] = get_image()
    image_buf[0] /= 255 # Normalization
    
    # Prediction
    im_model_output = im_model.predict([image_buf])

    output = im_model_output[0][0]

    if (im_model_output[0][0] < 0.125):
        im_action = 7

    elif (im_model_output[0][0] < 0.250):
        im_action = 6

    elif (im_model_output[0][0] < 0.375):
        im_action = 5

    elif (im_model_output[0][0] < 0.5):
        im_action = 8

    elif (im_model_output[0][0] < 0.625):
        im_action = 4

    elif (im_model_output[0][0] < 0.750):
        im_action = 3

    else:
        im_action = 2

    
    return im_action