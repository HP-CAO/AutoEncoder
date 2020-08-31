'''
 # @ Author: Hongi & Mirco
 # @ Create Time: 2020-08-04 10:20:57
 # @ Modified by: Your name
 # @ Modified time: 2020-08-04 10:21:26
 # @ Description: Training and heuristicly searching for counter examples in a smarter way
 '''

import os
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
import datetime
import numpy as np

from autoencoder import visulization as vis
from autoencoder import dataset
from autoencoder import config as cfg
from autoencoder import summary
from autoencoder.net import MLPAutoEncoder
from tensorflow.keras.layers import Input, Dense


class SpikeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SpikeLayer, self).__init__()

        self.spike_height = tf.Variable(initial_value = 1.0, trainable=True, shape=())

    def call(self, inputs, **kwargs):
        spike_value = self.spike_height + tf.sign(self.spike_height) * inputs[1]
        index_selection = tf.multiply(spike_value, tf.one_hot(inputs[2], input_length, dtype=tf.float32))

        return inputs[0] + tf.multiply(tf.cast(inputs[3], dtype=tf.float32), index_selection)


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

        remaining_inputs = [0.0, 0, False]

        for signals, labels in train_ds:
            with tf.GradientTape() as tape:
                train_error = fault_gen_model([signals] + remaining_inputs)
                gradients = tape.gradient(train_error, fault_gen_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, fault_gen_model.trainable_variables))
            train_loss_ave(train_error)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_ave.result(), step=step)

        # To evaluate on test dataset during training 
        for test_signals, test_labels in test_ds:
            test_error = fault_gen_model([test_signals] + remaining_inputs)
            test_loss_ave(test_error)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_ave.result(), step=step)

        # To evaluate on fault dataset during training
        for fault_signals, fault_labels in fault_ds:
            fault_error = fault_gen_model([fault_signals] + remaining_inputs)
            fault_loss_ave(fault_error)

        with fault_summary_writer.as_default():
            tf.summary.scalar('loss', fault_loss_ave.result(), step=step)

        template = 'Step:{}, Train_loss:{}, Test_loss:{}, Fault_loss:{}'
        print(template.format(step + 1,
                              train_loss_ave.result(),
                              test_loss_ave.result(),
                              fault_loss_ave.result()))

    fault_gen_model.save_weights(checkpoint_path)
    print('Training finished, model saved to {}'.format(checkpoint_path))


def search():
    steps = range(100)

    # load pre-trained weights
    weights_dir = f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir), \
        "The trained model not founded"
    weights_path = weights_dir + "/cp.ckpt"

    fault_gen_model.load_weights(weights_path)
    reconstruction_model.load_weights(weights_path)

    # samle a single signal for experiments 
    sample_index = np.random.randint(0, 300)
    sample = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))[sample_index]
    sample = dataset.add_noise(sample)
    sample = tf.cast(sample, dtype=tf.float32)
    sample = tf.reshape(sample, (1, 100))

    # reconstruct and  the original signal and visualization
    remaining_inputs = [0.0, 0, False]
    sample_reconstructed = reconstruction_model([sample] + remaining_inputs)
    error_ini = fault_gen_model([sample] + remaining_inputs)
    vis.ori_new_signals_plot(cfg.DATA_INPUT_DIMENSION, sample[0], sample_reconstructed[0])

    for i in range(input_length):

        # signal_base = np.copy(sample)
        signal_base = sample

        # initializing spike_height = 0
        spike_layer.spike_height.assign(0.001)

        for j in steps:
            # Optimizing spike phase

            with tf.GradientTape() as tape:
                error = fault_gen_model([signal_base, cfg.DATA_SPIKE_FAULT_MIN_VALUE, i, True])
                print("At {} th time step, searching steps {} ".format(i, j), error, spike_layer.spike_height)
                grads = tape.gradient(error, spike_layer.spike_height)
                optimizer.apply_gradients(zip([grads], [spike_layer.spike_height]))

        # error after optimizing 
        error = fault_gen_model([signal_base, cfg.DATA_SPIKE_FAULT_MIN_VALUE, i, True])
        #new_signal = spike_layer([signal_base, cfg.DATA_SPIKE_FAULT_MIN_VALUE, i, True])
        #error_for_new_signal = fault_gen_model([new_signal] + remaining_inputs)
        #vis.ori_new_signals_plot(cfg.DATA_INPUT_DIMENSION, signal_base[0], new_signal[0])
        
        print("===After searching, error:{}".format(error))

        if error < cfg.TEST_THRESHOLD:
            
            print("=================> COUNTERS FOUND")
            counter_example = spike_layer([signal_base, cfg.DATA_SPIKE_FAULT_MIN_VALUE, i, True])
            counter_error = fault_gen_model([counter_example] + remaining_inputs)
            
            print(error, counter_error)
            # Visulizing signal_base and counter_example
            vis.ori_new_signals_plot(cfg.DATA_INPUT_DIMENSION, signal_base[0], counter_example[0])
            
            #visualizing counter_example and its reconstruction
            counter_example_reconstructed = reconstruction_model([counter_example] + remaining_inputs)
            vis.ori_new_signals_plot(cfg.DATA_INPUT_DIMENSION, counter_example[0], counter_example_reconstructed[0])


if __name__ == "__main__":
    
    is_training = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # build the model 
    input_length = cfg.DATA_INPUT_DIMENSION
    
    signal_in = Input(shape=(input_length,), name='signal_in', dtype=tf.float32)
    #true_signal_in = Input(shape=(input_length,), name='true_signal_in', dtype=tf.float32)

    index_input = Input(shape=(), name="index_input", dtype=tf.int32)
    min_spike_height_input = Input(shape=(), name="min_spike_height_input", dtype=tf.float32)
    use_spike = Input(shape=(), name="use_spike", dtype=tf.bool)

    spike_layer = SpikeLayer()
    fault_signal = spike_layer([signal_in, min_spike_height_input, index_input, use_spike])
    dense1 = Dense(75, activation='relu', name='dense_1')(fault_signal)
    dense2 = Dense(50, activation='relu', name='dense_2')(dense1)
    dense3 = Dense(75, activation='relu', name='dense_3')(dense2)
    signal_out = Dense(100, activation=None, name='signal_out')(dense3)

    error_out = tf.keras.losses.MeanSquaredError()(fault_signal, signal_out)

    fault_gen_model = tf.keras.Model(inputs=[signal_in, min_spike_height_input, index_input, use_spike],
                                     outputs=error_out)

    fault_gen_model.summary()

    reconstruction_model = tf.keras.Model(inputs=[signal_in, min_spike_height_input, index_input, use_spike],
                                          outputs=signal_out)

    # define the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # time stamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # checkpoint_path
    checkpoint_path = './checkpoints/' + current_time + '/cp.ckpt'

    # computing graph
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fault", histogram_freq=100)
    tensorboard_callback.set_model(fault_gen_model)

    if is_training:
        train()
    else:
        search()
