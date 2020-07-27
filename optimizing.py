'''
 # @ Author: Hongi
 # @ Create Time: 2020-07-07 14:34:48
 # @ Modified by: Your name
 # @ Modified time: 2020-07-07 14:34:56
 # @ Description: To searching for adversarial examples based on gradient descend algorithm
 '''
import os
import tensorflow as tf
import numpy as np


from matplotlib import pyplot as plt
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg



if __name__ == "__main__":

    # build model    
    model = MLPAutoEncoder()
    
    # sample a single signal for experiments
    sample = dataset.data_normalization(np.loadtxt(cfg.TEST_FAULT_DATASET_PATH))[0]
    
    # add noise
    sample = dataset.add_noise(sample)

    # load pretraind weights
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    # convert input vector to trainable variables
    signal = [[x] for x in sample]
    signal  = tf.reshape(signal, (1, 100))
    signal = tf.Variable(signal, trainable=True)

    # define object function and optimizer
    optimize_object_fun = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    
    error = []
    episodes = range(500)

    for i in episodes:
        
        with tf.GradientTape() as tape:
            #tape.watch(signal)
            predictions = model(signal)
            optimize_error = optimize_object_fun(signal, predictions)
            gradients = tape.gradient(optimize_error, signal)
            optimizer.apply_gradients(zip([gradients], [signal]))
            error.append(optimize_error)

        print("Optimizing, current iter: {} <====> error:{}".format(i, optimize_error))
    
    signal_update = np.array(tf.reshape(signal,(100, 1)))

    # visualizing
    x = range(100)

    plt.subplot(2, 1, 1)
    plt.plot(x, sample, label='Origianl signal')
    plt.plot(x, signal_update, label='New signal')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(loc='upper right')


    plt.subplot(2, 1, 2)
    plt.plot(episodes, error)
    plt.xlabel('Searching iterations')
    plt.ylabel('Reconstruction Error')
    plt.show()






    

