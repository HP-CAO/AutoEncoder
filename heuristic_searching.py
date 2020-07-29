'''
 # @ Author: Hongi
 # @ Create Time: 2020-07-22 14:56:14
 # @ Modified by: Your name
 # @ Modified time: 2020-07-26 15:47:31
 # @ Description: Searching counter examples 
 '''

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense

from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg

if __name__ == "__main__":

    # build model
    # model = MLPAutoEncoder()

    # load pre-trained weights
    weights_dir = f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir), \
        "The trained model not founded"
    weights_path = weights_dir + "/cp.ckpt"
    # model.load_weights(weights_path)

    # samle a single signal for experiments 
    sample_index = np.random.randint(0, 300)
    sample = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))[sample_index]
    sample - dataset.add_noise(sample)
    sample = tf.cast(sample, dtype=tf.float32)
    sample = tf.reshape(sample, (100,))
    sample = np.array(sample)

    # define object function and optimizer
    optimize_object_fun = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    #############
    # Mircos code

    input_length = cfg.DATA_INPUT_DIMENSION

    signal_in = Input(shape=(input_length,), name='signal_in', dtype=tf.float32)

    dense1 = Dense(75, activation='relu', name="dense_1")(signal_in)
    dense2 = Dense(50, activation='relu', name="dense_2")(dense1)
    dense3 = Dense(75, activation='relu', name="dense_3")(dense2)
    signal_out = Dense(100, activation=None, name="dense_4")(dense3)

    in_out_model = tf.keras.Model(inputs=signal_in, outputs=signal_out)  # This is not really necessary

    error_out = tf.keras.losses.MeanSquaredError()(signal_in, signal_out)

    error_model = tf.keras.Model(inputs=signal_in, outputs=error_out)  # Train auto-encoder on this one

    spike_height = tf.Variable(initial_value=1.0, trainable=True, shape=())
    index_input = Input(shape=(), name="index_input", dtype=tf.int32)
    min_spike_height_input = Input(shape=(), name="min_spike_height_input", dtype=tf.float32)

    signed_min_spike_height = tf.multiply(tf.sign(spike_height), min_spike_height_input)

    spike_value = tf.add(spike_height, signed_min_spike_height)

    index_selection = tf.one_hot(index_input, input_length, on_value=spike_value)

    fault_signal = tf.add(signal_in, index_selection)

    fault_gen_model = tf.keras.Model(inputs=[signal_in, index_input, min_spike_height_input], outputs=fault_signal)


    ###########

    steps = range(10)
    counters = []
    error_model.load_weights(weights_path)

    for i in range(cfg.DATA_INPUT_DIMENSION):

        signal_base = np.copy(sample)

        for j in steps:

            # Training spike phase

            with tf.GradientTape as tape:
                error = error_model(fault_gen_model([signal_base, i, cfg.DATA_SPIKE_FAULT_MIN_VALUE]))
                print("While training", error)

            grads = tape.gradient(error, spike_height)
            optimizer.apply_gradients(zip(grads, spike_height))

        error = error_model(fault_gen_model([signal_base, i, cfg.DATA_SPIKE_FAULT_MIN_VALUE]))
        print("After training", error)


    # if len(counters) > 0:
    #     counters_file = open("counter_examples.txt", "w")
    #     for row in counters:
    #         np.savetxt(counters_file, row)



 #     episodes = range(500)
 #    counters = []
 #
 #    for i in range(cfg.DATA_INPUT_DIMENSION):
 #
 #        signal_base = np.copy(sample)
 #        signal_base[0][i] = signal_base[0][i] * cfg.DATA_SPIKE_FAULT_MIN_HEIGHT_RATIO
 #
 #        for j in episodes:
 #
 #            signal = tf.Variable(signal_base, trainable=True)
 #
 #            # SGD searching
 #            with tf.GradientTape() as tape:
 #                predictions = model(signal)
 #                optimizing_error = optimize_object_fun(signal, predictions)
 #
 #                if optimizing_error < cfg.TEST_THRESHOLD:
 #                    counter_example = {"index": i, "signal": signal}
 #                    counters.append(counter_example)
 #                    print("[Unsafe]: Counter_examples found, saved to counters lists")
 #
 #                gradients = tape.gradient(optimizing_error, signal)
 #                signal_base[0][i] = signal_base[0][i] - cfg.OPT_LEARNING_RATE * tf.abs(gradients[0][i])
 #
 #            print("Optimizing on time step {} , current iter:{} =======>Optimizing error: {}".format(i + 1, j + 1,
 #                                                                                                     optimizing_error))
 #
 #    if len(counters) > 0:
 #        counters_file = open("counter_examples.txt", "w")
 #        for row in counters:
 #            np.savetxt(counters_file, row)
