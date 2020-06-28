'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:03:53
 # @ Modified by: Your name
 # @ Modified time: 2020-06-28 14:09:45
 # @ Description: Defining variant autoencoders 
 '''

import tensorflow as tf
import random
import numpy as np
import os
import datetime

class MLPAutoEncoder(tf.keras.Model):
    
    def __init__(self):
        super(MLPAutoEncoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(75, activation='relu', input_shape=(100,))
        self.dense2 = tf.keras.layers.Dense(50, activation='relu')
        self.dense3 = tf.keras.layers.Dense(75, activation='relu')
        self.dense4 = tf.keras.layers.Dense(100, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs) 
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


class RNNAutoEncoder(tf.keras.Model):
    pass

class LstmAutoEncoder(tf.keras.Model):
    pass