import tensorflow as tf
import random
import numpy as np
import autoencoder.config as cfg

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model


# class MLPAutoEncoder(tf.keras.Model):
#
#     def __init__(self):
#         super(MLPAutoEncoder, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(75, activation='relu', input_shape=(100,))
#         self.dense2 = tf.keras.layers.Dense(50, activation='relu')
#         self.dense3 = tf.keras.layers.Dense(75, activation='relu')
#         self.dense4 = tf.keras.layers.Dense(100, activation=None)
#
#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         x = self.dense3(x)
#         return self.dense4(x)

class MLPAutoEncoder:

    def __init__(self):
        self.input_dim = 100
        self.dense1_dim = 75
        self.dense2_dim = 50
        self.dense3_dim = 75
        self.output_dim = 100
        self.activation_type = "relu"
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_signal = Input(shape=(self.input_dim,), dtype=tf.float32)
        dense1 = Dense(self.dense1_dim, activation=self.activation_type)(input_signal)
        dense2 = Dense(self.dense2_dim, activation=self.activation_type)(dense1)
        dense3 = Dense(self.dense3_dim, activation=self.activation_type)(dense2)
        output = Dense(self.output_dim, activation=None)(dense3)

        model = Model(inputs=input_signal, outputs=output)

        return model


# class MLPAutoEncoder(object):

#     def __init__(self):

#         # define layers 
#         self.input_length = cfg.DATA_INPUT_DIMENSION
#         self.signal_in = Input(shape=(self.input_length,), name='signal_in', dtype=tf.float32)
#         self.dense_1 = Dense(75, activation='relu', name='dense_1')(self.signal_in)
#         self.dense_2 = Dense(50, activation='relu', name='dense_2')(self.dense_1)
#         self.dense_3 = Dense(75, activation='relu', name='dense_3')(self.dense_2)
#         self.signal_out = Dense(100, activation='relu', name='dense_4')(self.dense_3)
#         self.error_out = tf.keras.losses.MeanSquaredError()(self.signal_in, self.signal_out)

#         # define spike_fault
#         self.spike_height = tf.Variable(initial_value=1.0, trainable=True, shape=())
#         self.index_input = Input(shape=(), name='index_input', dtype=tf.int32)
#         self.min_spike_height_input = Input(shape=(), name='min_spike_height_input', dtype=tf.float32)
#         self.signed_min_spike_height = tf.multiply(tf.sign(self.spike_height), self.min_spike_height_input)
#         self.spike_value = tf.add(self.spike_height, self.signed_min_spike_height)
#         self.index_selection = tf.one_hot(self.index_input, self.input_length, on_value = self.spike_value)
#         self.fault_signal = tf.add(self.signal_in, self.index_selection)

#     def in_out_model(self):
#         model = tf.keras.Model(inputs=self.signal_in, outputs=self.signal_out)
#         return model

#     def error_model(self):
#         model = tf.keras.Model(inputs=self.signal_in, outputs=self.error_out)
#         return model

#     def fault_gen_model(self):
#         model = tf.keras.Model(inputs=[self.signal_in, self.index_input, self.min_spike_height_input], outputs = self.fault_signal)
#         return model

