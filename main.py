'''
 # @ Author: Hongi
 # @ Create Time: 2020-05-14 14:17:14
 # @ Modified by: Your name
 # @ Modified time: 2020-05-14 14:31:44
 # @ Description: Model training and evaluation 
 '''


import tensorflow as tf
import random
import numpy as np
import os
import datetime

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model  
from tensorflow.keras import losses
from tensorflow.keras import optimizers

class AutoEncoder(Model):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.dense1 = Dense(25, activation=tf.nn.tanh, input_shape=(50,))
        self.dense2 = Dense(10, activation=tf.nn.tanh)
        self.dense3 = Dense(25, activation=tf.nn.tanh)
        self.dense4 = Dense(50, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs) 
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

def load_data():
## Data_dir checking  
    try:
        os.path.exists('train.txt')
        os.path.exists('test.txt')
        os.path.exists('fault.txt')
    except FileNotFoundError:
        print("Have you ever created train/tes/fault dataset?")
    
## Load training data 
    train_data = np.loadtxt('train.txt')
    x_train = train_data
    y_train = train_data
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(64)

## load_ testing data
    test_data = np.loadtxt('test.txt')
    x_test = test_data
    y_test = test_data
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(64)

## laod_fault data
    fault_data = np.loadtxt('fault.txt')
    x_fault = fault_data
    y_fault = fault_data
    fault_ds = tf.data.Dataset.from_tensor_slices(
        (x_fault, y_fault)).batch(64)
    
    print("----------Data loaded!----------")
    return train_ds, test_ds, fault_ds
    
def train(signal, labels):
    with tf.GradientTape() as tape:
        predictions = model(signal)
        loss = loss_object(labels,predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

def test(signal, labels):
    predictions = model(signal)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)

def fault_test(signal, labels):
    """ For fault_signal test while training """
    predictions = model(signal)
    f_loss = loss_object(labels, predictions)
    fault_loss(f_loss)


if __name__ == "__main__":
    
    ## Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    fault_log_dir = 'logs/gradient_tape/' + current_time + '/fault'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    fault_summary_writer = tf.summary.create_file_writer(fault_log_dir)

    ## Set up checkpoint
    checkpoint_path = './training_track/'+ current_time +'/cp.ckpt'
    #checkpoint_dir = os.path.dirname(checkpoint_path)


    EPOCHS = 300
    loss_object = losses.MeanSquaredError()  
    optimizer = optimizers.Adam()

    model = AutoEncoder()
    train_ds, test_ds, fault_ds = load_data()

    ## To compute the average loss for all data in each epoch

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    fault_loss = tf.keras.metrics.Mean(name='fault_loss')

    for epoch in range(EPOCHS):

        train_loss.reset_states()
        test_loss.reset_states()

        for signals, labels in train_ds:
            train(signals, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            
        for test_signals, test_labels in test_ds:
            test(test_signals, test_labels)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)

        for fault_signals, fault_labels in fault_ds:
            fault_test(fault_signals, fault_labels)
        with fault_summary_writer.as_default():
            tf.summary.scalar('loss', fault_loss.result(), step=epoch)
    
        template = 'Epoch{}, Train_loss:{}, Test_loss:{}, Fault_loss:{}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            test_loss.result(),
                            fault_loss.result()))

    
    model.save_weights(checkpoint_path)
    print(f" Training finished, model saved to {checkpoint_path}")
