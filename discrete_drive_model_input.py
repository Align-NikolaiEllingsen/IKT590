import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.models import load_model
import sys
import time
import numpy as np

import airsim

import keras.backend as K
from keras.preprocessing import image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Trained model path
MODEL_PATH = './im_model//model_rgb.h5'

model = load_model(MODEL_PATH)

# Connect to AirSim 
client = airsim.CarClient()
client.confirmConnection()

# Initialize image buffer
image_buf = np.zeros((1, 66, 200, 3))

def get_image():
    """
    Get image from AirSim client
    """
    image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    try:
        image_rgb = image1d.reshape(image_response.height, image_response.width, 3)
    except: 
        image_rgb = image1d.reshape(image_response.height, image_response.width, 3)
    return image_rgb[78:144,27:227,0:3].astype(float)

def get_im_response():    
    # Update throttle value according to steering angle
    image_buf[0] = get_image()
    image_buf[0] /= 255 # Normalization
    
    # Prediction
    model_output = model.predict([image_buf])

    end_time = time.time()
    received_output = model_output[0][0]

    if (model_output[0][0] < 0.1):
        im_action = 6

    elif (model_output[0][0] < 0.2):
        im_action = 5

    elif (model_output[0][0] < 0.4):
        im_action = 4

    elif (model_output[0][0] < 0.5):
        im_action = 4

    elif (model_output[0][0] < 0.6):
        im_action = 3

    elif (model_output[0][0] < 0.8):
        im_action = 2
    
    else:
        im_action = 2
    
    return im_action