'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:21:22
 # @ Modified by: Your name
 # @ Modified time: 2020-06-28 15:26:28
 # @ Description: Training scripts
 '''
 
import tensorflow as tf
import random
import numpy as np
import os
import datetime
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg
from autoencoder import summary


def train():
    
    for epoch in range(cfg.TRAIN_EPOCHS):

        # Reset the training loss before each episode
        train_loss.reset_states()
        test_loss.reset_states()
        fault_loss.reset_states()
        
        for signals, labels in train_ds:   
            with tf.GradientTape() as tape:
                train_predictions = model(signals)
                trainloss = loss_object(labels, train_predictions)
                gradients = tape.gradient(trainloss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(trainloss)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
        
        # To evaluate the autoecoder on test dataset during running time in case overfitting
        for test_signals, test_labels in test_ds:
            test_predictions = model(test_signals)
            testloss = loss_object(test_labels, test_predictions)
            test_loss(testloss)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)

        # To evaluate the autoecoder on fault dataset during running time in case overfitting
        for fault_signals, fault_labels in fault_ds:
            fault_predictions = model(fault_signals)
            faultloss = loss_object(fault_labels, fault_predictions)
            fault_loss(faultloss)

        with fault_summary_writer.as_default():
            tf.summary.scalar('loss', fault_loss.result(), step=epoch)
    
        template = 'Epoch{}, Train_loss:{}, Test_loss:{}, Fault_loss:{}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            test_loss.result(),
                            fault_loss.result()))

    
    model.save_weights(checkpoint_path)
    print(f" Training finished, model saved to {checkpoint_path}")



if __name__ == "__main__":


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set up checkpoint
    checkpoint_path = './checkpoints/'+ current_time + '/cp.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Setting training configuration
    loss_object = tf.keras.losses.MeanSquaredError()  
    optimizer = tf.keras.optimizers.Adam()
    model = MLPAutoEncoder()

    # Load training data
    train_ds = dataset.load_data(cfg.TRAIN_DATASET_PATH)
    test_ds = dataset.load_data(cfg.TEST_DATASET_PATH)
    fault_ds = dataset.load_data(cfg.TEST_FAULT_DATASET_PATH)

    # To compute the average loss for all data in each epoch
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    fault_loss = tf.keras.metrics.Mean(name='fault_loss')

    # Build summary writer for visulizing loss curves 
    train_summary_writer, test_summary_writer, fault_summary_writer = summary.build_summary_writer(current_time)

    train()


