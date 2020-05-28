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
MODEL_PATH = './models/fresh_models/model.h5'

model = load_model(MODEL_PATH)

# Connect to AirSim 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# Start driving
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
client.setCarControls(car_controls)

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

while True:    
    # Update throttle value according to steering angle
    if abs(car_controls.steering) <= 1.0:
        car_controls.throttle = 0.4
    else:
        car_controls.throttle = 0.5
    
    image_buf[0] = get_image()
    image_buf[0] /= 255 # Normalization

    start_time = time.time()
    
    # Prediction
    model_output = model.predict([image_buf])

    end_time = time.time()
    received_output = model_output[0][0]

    # Rescale prediction to [-1,1] and factor by 0.82 for drive smoothness

    #car_controls.steering = round((0.82*(float((model_output[0][0]*2.0)-1))), 2)

    if (model_output[0][0] < 0.1):
        car_controls.steering = -1

    elif (model_output[0][0] < 0.2):
        car_controls.steering = -0.5

    elif (model_output[0][0] < 0.4):
        car_controls.steering = -0.25

    elif (model_output[0][0] < 0.5):
        car_controls.steering = 0

    elif (model_output[0][0] < 0.6):
        car_controls.steering = 0.25

    elif (model_output[0][0] < 0.8):
        car_controls.steering = 0.5

    else:
        car_controls.steering = -1

    
    # Print progress
    print('Sending steering = {0}, throttle = {1}, prediction time = {2}'.format(model_output[0][0], car_controls.throttle,str(end_time-start_time)))
    
    # Update next car state
    client.setCarControls(car_controls)
    
    # Wait a bit between iterations
    time.sleep(0.05)


client.enableApiControl(False)