'''
 # @ Author: Hongi & Mirco
 # @ Create Time: 2020-08-04 10:20:57
 # @ Modified by: Your name
 # @ Modified time: 2020-08-04 10:21:26
 # @ Description: Training and heuristicly searching for counter examples in a smarter way
 '''

import os
import tensorflow as tf
import datetime
import numpy as np

from autoencoder import dataset
from autoencoder import config as cfg
from autoencoder import summary
from autoencoder.net import MLPAutoEncoder
from tensorflow.keras.layers import Input, Dense    


def train():
    
    # load the training data
    train_ds = dataset.load_data(cfg.TRAIN_DATASET_PATH)
    test_ds = dataset.load_data(cfg.TEST_DATASET_PATH)
    fault_ds = dataset.load_data(cfg.TEST_FAULT_DATASET_PATH)


    # build summaru writer for losses curve
    train_summary_writer, test_summary_writer, fault_summary_writer = summary.build_summary_writer(current_time)

    # build losses averaging operator
    train_loss_ave = tf.keras.metrics.Mean(name='train_loss_ave')
    test_loss_ave = tf.keras.metrics.Mean(name='test_loss_ave')
    fault_loss_ave = tf.keras.metrics.Mean(name='fault_loss_ave')


    for step in range(cfg.TRAIN_EPOCHS):
        # reset the training loss at each step

        train_loss_ave.reset_states()
        test_loss_ave.reset_states()
        fault_loss_ave.reset_states()

        for signals, labels in train_ds:
            with tf.GradientTape() as tape:
                train_error = error_model(signals)
                gradients  = tape.gradient(train_error, error_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, error_model.trainable_variables))
            train_loss_ave(train_error)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_ave.result(), step=step)

        # To evaluate on test dataset during training 
        for test_signals, test_labels in test_ds:
            test_error = error_model(test_signals)
            test_loss_ave(test_error)
        
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_ave.result(), step=step)

        
        # To evaluate on fault dataset during training
        for fault_signals, fault_labels in fault_ds:
            fault_error = error_model(fault_signals)
            fault_loss_ave(fault_error)
        
        with fault_summary_writer.as_default():
            tf.summary.scalar('loss', fault_loss_ave.result(), step=step)

        template = 'Step:{}, Train_loss:{}, Test_loss:{}, Fault_loss:{}'
        print(template.format(step + 1,
                            train_loss_ave.result(),
                            test_loss_ave.result(),
                            fault_loss_ave.result()))
    
    error_model.save_weights(checkpoint_path)
    print('Training finished, model saved to {}'.format(checkpoint_path))            


def search():

    steps = range(10)

    # load pre-trained weights
    weights_dir = f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir), \
        "The trained model not founded"
    weights_path = weights_dir + "/cp.ckpt"
    error_model.load_weights(weights_path)

    # samle a single signal for experiments 
    sample_index = np.random.randint(0, 300)
    sample = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))[sample_index]
    sample = dataset.add_noise(sample)
    sample = tf.cast(sample, dtype=tf.float32)
    sample = tf.reshape(sample, (1, 100))

    for i in range(input_length):
       
        #signal_base = np.copy(sample)
        signal_base = sample

        for j in steps:

            # Training spike phase

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(spike_height)
                
                error = error_model(fault_gen_model([signal_base, i, cfg.DATA_SPIKE_FAULT_MIN_VALUE]))
                print("While training", error, spike_height)
                print([var.name for var in tape.watched_variables()])
                #grads = tape.gradient(error, error_model.trainable_variables)
                
                grads = tape.gradient(error, spike_height)
                optimizer.apply_gradients(zip([grads], [spike_height]))

        error = error_model(fault_gen_model([signal_base, i, cfg.DATA_SPIKE_FAULT_MIN_VALUE]))
        print("After training", error)


if __name__ == "__main__":


    # build the model 
    input_length = cfg.DATA_INPUT_DIMENSION

    signal_in = Input(shape=(input_length,), name='signal_in', dtype=tf.float32)

    dense1 = Dense(75, activation='relu', name='dense_1')(signal_in)
    dense2 = Dense(50, activation='relu', name='dense_2')(dense1)
    dense3 = Dense(75, activation='relu', name='dense_3')(dense2)
    signal_out = Dense(100, activation=None, name='signal_out')(dense3)

    error_out = tf.keras.losses.MeanSquaredError()(signal_in, signal_out)

    error_model = tf.keras.Model(inputs=signal_in, outputs=error_out)

    spike_height = tf.Variable(initial_value=1.0, trainable=True, shape=())
    index_input = Input(shape=(), name="index_input", dtype=tf.int32)
    min_spike_height_input = Input(shape=(), name="min_spike_height_input", dtype=tf.float32)

    signed_min_spike_height = tf.multiply(tf.sign(spike_height), min_spike_height_input)

    spike_value = tf.add(spike_height, signed_min_spike_height)

    index_selection = tf.one_hot(index_input, input_length, on_value=spike_value)

    fault_signal = tf.add(signal_in, index_selection)
    
    fault_gen_model = tf.keras.Model(inputs=[signal_in, index_input, min_spike_height_input], outputs=fault_signal)

    # define the optimizer
    optimizer = tf.keras.optimizers.Adam()
    
    # time stamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # checkpoint_path
    checkpoint_path = './checkpoints/'+ current_time + '/cp.ckpt'

    is_trianing = False
    
    if is_trianing:
        train()
    else:
        search()