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
import autoencoder.config as cfg

from tensorflow.keras.layers import Dense, Input

# class MLPAutoEncoder(object):
    
#     def __init__(self):
#         super(MLPAutoEncoder, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(75, activation='relu', input_shape=(100,))
#         self.dense2 = tf.keras.layers.Dense(50, activation='relu')
#         self.dense3 = tf.keras.layers.Dense(75, activation='relu')
#         self.dense4 = tf.keras.layers.Dense(100, activation=None)

#     def call(self, inputs):
#         x = self.dense1(inputs) 
#         x = self.dense2(x)
#         x = self.dense3(x)
#         return self.dense4(x)

class MLPAutoEncoder(object):

    def __init__(self):

        # define layers 
        self.input_length = cfg.DATA_INPUT_DIMENSION
        self.signal_in = Input(shape=(self.input_length,), name='signal_in', dtype=tf.float32)
        self.dense_1 = Dense(75, activation='relu', name='dense_1')(self.signal_in)
        self.dense_2 = Dense(50, activation='relu', name='dense_2')(self.dense_1)
        self.dense_3 = Dense(75, activation='relu', name='dense_3')(self.dense_2)
        self.signal_out = Dense(100, activation='relu', name='dense_4')(self.dense_3)
        self.error_out = tf.keras.losses.MeanSquaredError()(self.signal_in, self.signal_out)

        # define spike_fault
        self.spike_height = tf.Variable(initial_value=1.0, trainable=True, shape=())
        self.index_input = Input(shape=(), name='index_input', dtype=tf.int32)
        self.min_spike_height_input = Input(shape=(), name='min_spike_height_input', dtype=tf.float32)
        self.signed_min_spike_height = tf.multiply(tf.sign(self.spike_height), self.min_spike_height_input)
        self.spike_value = tf.add(self.spike_height, self.signed_min_spike_height)
        self.index_selection = tf.one_hot(self.index_input, self.input_length, on_value = self.spike_value)
        self.fault_signal = tf.add(self.signal_in, self.index_selection)
    
    def in_out_model(self):
        model = tf.keras.Model(inputs=self.signal_in, outputs=self.signal_out)
        return model
        
    def error_model(self):
        model = tf.keras.Model(inputs=self.signal_in, outputs=self.error_out)
        return model
    
    def fault_gen_model(self):
        model = tf.keras.Model(inputs=[self.signal_in, self.index_input, self.min_spike_height_input], outputs = self.fault_signal)
        return model

class RNNAutoEncoder(tf.keras.Model):
    pass

class LstmAutoEncoder(tf.keras.Model):
    pass