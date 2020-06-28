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


if __name__ == "__main__":
    

    training_time_stamp = '20200624-192943'
    weights_dir =  f"./training_track/{training_time_stamp}"
        
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  

    weights_path = weights_dir +"/cp.ckpt"
    
    model = AutoEncoder()
    model.load_weights(weights_path)

    x_test, x_fault = load_demo_data()
    test_pieces, fault_pieces = data_pick(x_test, x_fault) ## Pick 5 samples from test_data and fault_data each
    x=np.arange(100)

    test_pieces = np.array(test_pieces)
    fault_pieces = np.array(fault_pieces)

    test_predictions = model.predict(test_pieces)
    fault_predictions = model.predict(fault_pieces)

    ## visualize signal and signal reconstructed
    for i in np.arange(len(test_pieces)):

        plt.plot(x, test_pieces[i])
        plt.plot(x, test_predictions[i])  
        plt.show()
        
    for i in range(len(fault_pieces)):
        
        plt.plot(x, fault_pieces[i])
        plt.plot(x, fault_predictions[i])
        plt.show()

    
