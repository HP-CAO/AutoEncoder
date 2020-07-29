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

from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg

if __name__ == "__main__":

    # build model
    model = MLPAutoEncoder()

    # load pre-trained weights
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    # samle a single signal for experiments 
    sample_index = np.random.randint(0, 300)
    sample = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))[sample_index]
    sample - dataset.add_noise(sample)
    sample = tf.cast(sample, dtype=tf.float32)
    sample = tf.reshape(sample, (1, 100))
    sample = np.array(sample)

    # define object function and optimizer
    optimize_object_fun = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    episodes = range(500)
    counters = []

    for i in range(cfg.DATA_INPUT_DIMENSION):       

        signal_base = np.copy(sample)  
        signal_base[0][i] = signal_base[0][i] * cfg.DATA_SPIKE_FAULT_MIN_HEIGHT_RATIO

        for j in episodes:            

            signal = tf.Variable(signal_base, trainable=True)
            
            # SGD searching
            with tf.GradientTape() as tape:
                predictions = model(signal)
                optimizing_error = optimize_object_fun(signal, predictions)

                if optimizing_error < cfg.TEST_THRESHOLD:
                    counter_example = {"index":i, "signal":signal}
                    counters.append(counter_example)
                    print("[Unsafe]: Counter_examples found, saved to counters lists")

                gradients = tape.gradient(optimizing_error, signal)
                signal_base[0][i] = signal_base[0][i] - cfg.OPT_LEARNING_RATE * tf.abs(gradients[0][i])
                                
            print("Optimizing on time step {} , current iter:{} =======>Optimizing error: {}".format(i + 1 ,j + 1, optimizing_error))
    
            
    if len(counters)>0:
        counters_file = open("counter_examples.txt", "w")
        for row in counters:
            np.savetxt(counters_file, row)


                
            
            
            

