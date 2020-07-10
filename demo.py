'''
 # @ Author: Hongi
 # @ Create Time: 2020-05-14 14:17:14
 # @ Modified by: Hongi
 # @ Modified time: 2020-05-14 14:26:58
 # @ Description: Demo for signal reconstruction and visualization 
 '''

import os
import numpy as np
import tensorflow as tf
import datetime 
from matplotlib import pyplot as plt
from autoencoder import config as cfg
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset



if __name__ == "__main__":
    
    # Build Model 
    model = MLPAutoEncoder()
        
    # Load pre-trained weights 
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    # Load test-cases(normal and fault signal)
    x_test = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))
    x_fault = dataset.data_normalization(np.loadtxt(cfg.TEST_FAULT_DATASET_PATH))

    # Randomly sample from dataset
    test_pieces, fault_pieces = dataset.data_pick(x_test, x_fault, cfg.TEST_SAMPLE_NUM)
    
    # Add gaussain noise
    if cfg.DATA_ADD_NOISE:
        test_pieces = dataset.add_noise(test_pieces)
        fault_pieces = dataset.add_noise(fault_pieces)

    # Convert to np.array for batch test
    test_pieces = np.array(test_pieces)
    fault_pieces = np.array(fault_pieces)

    # Do inference
    test_predictions = model.predict(test_pieces)
    fault_predictions = model.predict(fault_pieces)


    # visualization
    x=np.arange(100)
    for i in range(cfg.TEST_SAMPLE_NUM):

        # subplot_1 for normal test cases
        plt.subplot(3, 1 ,1)
        plt.plot(x, test_pieces[i], label='Normal signal')
        plt.plot(x, test_predictions[i], label='Reconstructed normal signal')
        plt.legend(loc='upper right')
        plt.title('Normal signal')

        # subplot_2 for abnormal test cases
        plt.subplot(3, 1 ,2)
        plt.plot(x, fault_pieces[i], label='Abnormal signal')
        plt.plot(x, fault_predictions[i], label='Reconstructed abnormal signal')
        plt.legend(loc='upper right')
        plt.ylabel('magnitude:x(t)')
        plt.title('Abnormal signal')

        # subplot_3 for comparation between normal/abnormal reconstruction
        plt.subplot(3, 1 ,3)
        plt.plot(x, test_predictions[i], label='Normal signal constructed')
        plt.plot(x, fault_predictions[i], label='Abnormal signal constructed')
        plt.legend(loc='upper right')
        plt.xlabel('time:t')
        plt.title('Reconstruction Comparation')

        plt.show()
        


    
