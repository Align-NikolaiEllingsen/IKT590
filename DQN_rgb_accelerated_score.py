#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import threading
from collections import deque
import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import tensorflow as tf
import sys
import csv
import keyboard
import airsim
import time
import im_replay

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hides useless log print

# Connect to AirSim 
client = airsim.CarClient(ip="127.0.0.2")
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
car_state = client.getCarState() 

# Start driving
action_index = 0

episodes = 0 
old_episode = 0
log_reward = 0
terminal = False

reward = 0
episode_reward = 0
alive_reward = 0.1
r_t = 0

data_buf = np.zeros((2))

use_checkpoints = 0

ACTIONS = 9
GAMMA = 0.9
LEARNING_RATE = 0.0001 
OBSERVE = 0
IMITATE = 25000
EXPLORE = 100000
TRAIN = 100000
FINAL_EPSILON = 0.01 
INITIAL_EPSILON = 0.5
REPLAY_MEMORY = 10000
BATCH = 32 # Make process batch after observe for adqn
FRAME_PER_ACTION = 1

filename = "./logs/log_file_score.csv"


throttle = 1
brake = 0
steering_angle = 0
handbrake = 0

data_collect = []

image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
image_bgr = image1d.reshape(image_response.height, image_response.width, 3)
#image_grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
b,g,r = cv2.split(image_bgr)
x_t = cv2.merge((r,g,b))
x_t = x_t[40:101,36:220,0:3].astype(float)

image_buf = x_t
image_buf /= 255 # Normalization

s_t = image_buf#np.stack((x_t, x_t, x_t, x_t), axis=2)
s_t1=s_t
im_action = 1

#s_tlm = data_buf
#s_tlm1 = s_tlm

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])

    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])

    W_conv3 = weight_variable([5, 5, 36, 48])
    b_conv3 = bias_variable([48])

    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])

    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])

    #W_fc0 = weight_variable([11776, 1164])
    #b_fc0 = bias_variable([1164])

    #W_fc1 = weight_variable([11776, 512])
    #b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([11776, 512])
    b_fc2 = bias_variable([512])

    W_fc3 = weight_variable([512, 256])
    b_fc3 = bias_variable([256])

    W_fc4 = weight_variable([256, ACTIONS])
    b_fc4 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 61, 184, 3])
    #s2 = tf.placeholder("float", [None, 2])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)

    #h_pool1 = max_pool_2x2(h_conv1)
    

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

    #h_pool3 = max_pool_2x2(h_conv3)

    h_drop1 = tf.nn.dropout(h_conv3, 0.5)

    h_conv4 = tf.nn.relu(conv2d(h_drop1, W_conv4, 1) + b_conv4)

    #h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

    h_conv5_flat = tf.reshape(h_conv5, (-1, 11776))

    #h_tlm = tf.concat([h_conv5_flat, s2], 1)

    #h_fc0 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc0) + b_fc0)

    #h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    h_fc2 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc2) + b_fc2)

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # readout layer
    readout = tf.matmul(h_fc3, W_fc4) + b_fc4
    
    #return s, s2, readout, h_fc3
    return s, readout, h_fc3

def log_results(filename, data_collect):
    # Save the results to a file so we can graph it later.
    global OBSERVE
    global EXPLORE
    global TRAIN
    while True:
        try:      
            with open(filename, 'w', newline='') as data_dump:
                wr = csv.writer(data_dump)
                wr.writerows(data_collect)
            break
        except:
            print("CSV Error")

#def trainNetwork(s, s2, readout, h_fc1, sess):
def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    D = deque()
    #s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks_score")

    
    if checkpoint and checkpoint.model_checkpoint_path and use_checkpoints:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    global s_t
    global r_t
    global reward
    global s_tlm
    global terminal
    global action_index
    global im_action
    #global time_image
    global episode_reward
    global episodes
    global log_reward
    global BATCH
    global old_episode
    while True:
        #time_start = time.time()
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        #readout_t = readout.eval(feed_dict={s : [s_t], s2 : [s_tlm]})[0]
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            #if random.random() <= epsilon or t <= OBSERVE:
            #    if random.random() <= 0.5 or t <= OBSERVE:
            #        action_index = im_action
            #        a_t[im_action] = 1
            #        selector = "IM"
            #    else:
            #        action_index = random.randrange(ACTIONS)
            #        a_t[random.randrange(ACTIONS)] = 1
            #        selector = "RND"
            random_select = random.random()

            #if t <= OBSERVE:
            #    action_index = random.randrange(ACTIONS)
            #    a_t[random.randrange(ACTIONS)] = 1
            #    selector = "RND"

            if t <= OBSERVE + IMITATE or random_select <= (epsilon/2):
                action_index = im_action
                a_t[im_action] = 1
                selector = "IM"

            elif t <= OBSERVE or random_select <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
                selector = "RND"

            else:
                action_index = np.argmax(readout_t)
                #print (readout_t)
                a_t[action_index] = 1
                selector = "DQN"

            #if action_index == im_action:
            #    episode_reward += 0.01
                    
            r_t = episode_reward
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (EXPLORE + IMITATE)

        s_t1 = image_buf#np.append(x_t, s_t[:, :, :3], axis=2)
        #s_tlm1 = s_tlm

        #D.append((s_t, s_tlm, a_t, r_t, s_t1, s_tlm1, terminal))
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        #if t > OBSERVE + EXPLORE:
        #    BATCH = 1
        #if t >= (OBSERVE + IMITATE + EXPLORE + TRAIN):
        #    saver.save(sess, 'saved_networks_score/' 'checkpoint', global_step = t)
        #    break
        #    break
        if  t > BATCH:#t > OBSERVE: #and t % 100 == 0: #True:#t > OBSERVE: #and
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            # get the batch variables
            """
            s_j_batch = [d[0] for d in minibatch]
            s2_j_batch = [d[1] for d in minibatch]
            a_batch = [d[2] for d in minibatch]
            r_batch = [d[3] for d in minibatch]
            s_j1_batch = [d[4] for d in minibatch]
            s2_j1_batch = [d[5] for d in minibatch]
            """
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            
            y_batch = []
            #readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch, s2 : s2_j1_batch})
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}#,
                #s2 : s2_j_batch}
            )

        # update
        s_t = s_t1
        #s_tlm = s_tlm1
        t += 1

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + IMITATE:
            state = "imitate"
        elif t > IMITATE and t <= OBSERVE + IMITATE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state, \
        "/ EPSILON", round(epsilon,4), "/ ACTION", action_index, selector, "/ REWARD", round(reward,4), "/EPISODE REWARD", round(episode_reward,4), \
        "/ Q_MAX %e" % round(np.max(readout_t),4), "\EPISODE", episodes)
        if t % 100000 == 0:
            saver.save(sess, 'saved_networks_score/' 'checkpoint', global_step = t)

        log_reward += episode_reward

        if t % 1000 == 0:
            data_collect.append([t, action_index, selector, reward, episode_reward, log_reward, np.max(readout_t), episodes])
            log_results(filename, data_collect)
            log_reward = 0

def interpret_action(action_index):
    # A dictionary would obviously be better! However, due to the length, i select the sub-optimal elif spam
    #action = {0:car_controls.throttle = 1, 1:car_controls.brake = 1, 2:car_controls.steering = 1, 3:car_controls.steering = 0.9, 4:car_controls.steering = 0.8, 5: car_controls.steering = 0.7, 6: car_controls.steering = 0.6}
    car_controls.throttle = 0.4
    car_controls.brake = 0
    car_controls.steering = 0
    #if action_index == 0:
    #    car_controls.throttle = 1
    #    car_controls.brake = 0
    #elif action_index == 1:
    #    car_controls.throttle = 0
    #    car_controls.brake = 1
    if action_index == 0:
        car_controls.steering = 1
    elif action_index == 1:
        car_controls.steering = 0.5
    elif action_index == 2:
        car_controls.steering = 0.25
    elif action_index == 3:
        car_controls.steering = -0.25
    elif action_index == 4:
        car_controls.steering = -0.5
    elif action_index == 5:
        car_controls.steering = -1
    else:
        car_controls.steering = 0
    return car_controls


def interpret_action_throttle(action_index):
    # A dictionary would obviously be better! However, due to the length, i select the sub-optimal elif spam
    #action = {0:car_controls.throttle = 1, 1:car_controls.brake = 1, 2:car_controls.steering = 1, 3:car_controls.steering = 0.9, 4:car_controls.steering = 0.8, 5: car_controls.steering = 0.7, 6: car_controls.steering = 0.6}
    car_controls.throttle = 0.4
    car_controls.brake = 0
    car_controls.steering = 0
    if action_index == 0:
        car_controls.throttle = 1
        car_controls.brake = 0
    elif action_index == 1:
        car_controls.throttle = 0
        car_controls.brake = 1
    if action_index == 2:
        car_controls.steering = 1
    elif action_index == 3:
        car_controls.steering = 0.5
    elif action_index == 4:
        car_controls.steering = 0.25
    elif action_index == 5:
        car_controls.steering = -0.25
    elif action_index == 6:
        car_controls.steering = -0.5
    elif action_index == 7:
        car_controls.steering = -1
    else:
        car_controls.steering = 0
    return car_controls

def telemetry():
    while True:
        global x_t
        global reward
        global episode_reward
        global terminal
        global im_action
        #global time_train
        #global time_total
        #global time_image
        global episodes
        while True:
            try:
                #client.simPause(True)
                image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
                image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
                image_bgr = image1d.reshape(image_response.height, image_response.width, 3)
                #image_grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                b,g,r = cv2.split(image_bgr)
                image_rgb = cv2.merge((r,g,b))
                x_t = image_rgb[40:101,36:220,0:3].astype(float)
                image_buf = x_t
                image_buf /= 255 # Normalization

                #from PIL import Image
                #img = Image.fromarray(image_rgb[40:101,36:220,0:3])
                #img.save('my.png')
                #img.show()
                #x_t = np.reshape(x_t, (61, 184, 1))
                #client.simContinueForTime(0.2)
                break
            except:
                print("Image Error")
        car_state = client.getCarState()
        #alive_reward += 5
        #print (s_t1) 
        reward = max(0,(car_state.speed))/33#+ alive_reward
        episode_reward += reward
        #s_tlm = round(abs(car_state.speed),1), round(car_state.kinematics_estimated.linear_acceleration.y_val,1)

        #client.simPause(False)
        car_controls = interpret_action(action_index)
        client.setCarControls(car_controls)
        collision_info = client.simGetCollisionInfo()  
        im_action = im_replay.get_im_response()
        #print (car_state.kinematics_estimated.linear_velocity.x_val)  
        #?????? fix crash when map change
        
        terminal = False
        if collision_info.has_collided:
            reward = -1
            episode_reward = 0
            terminal = True
            client.reset()
            episodes += 1
        if car_state.kinematics_estimated.position.z_val <= -1:
            client.reset()

def loop():
    sess = tf.InteractiveSession()
    #s, s2, readout, h_fc1 = createNetwork()
    #trainNetwork(s, s2, readout, h_fc1, sess)
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

t = threading.Thread(target=loop, name='TrainingLoop')
t2 = threading.Thread(target=telemetry, name='TelemetryLoop')

t.start()
t2.start()

t.join()
t2.join()