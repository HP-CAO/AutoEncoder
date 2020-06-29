'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-24 17:16:55
 # @ Modified by: Your name
 # @ Modified time: 2020-06-24 17:17:02
 # @ Description: Test and compute reconstruction error for each sample 
 '''

import os
import numpy as np
import tensorflow as tf
from autoencoder.net import MLPAutoEncoder
from matplotlib import pyplot as plt

def data_normalization(data_set):
    """Do gloabal normalization for data"""
    # (xi-Xmin)/(Xmax-min)
    
    data_set = (data_set-np.min(data_set))/(np.max(data_set)-np.min(data_set))

    return data_set


if __name__ == "__main__":

    training_time_stamp = '20200624-192943'
    weights_dir =  f"./training_track/{training_time_stamp}"
    weights_path = weights_dir +"/cp.ckpt"


    model = MLPAutoEncoder()
    model.load_weights(weights_path)

    test_data = np.array(np.loadtxt('speed_test.txt'))
    fault_data =np.array(np.loadtxt('spike_fault.txt'))

    x_test = data_normalization(test_data)
    x_fault = data_normalization(fault_data)

    x_test_prediction = model.predict(x_test)
    x_fault_prediction = model.predict(x_fault)

    test_error = []
    fault_error = [] 

    for i in np.arange(len(x_test)):
        error = np.sum(np.square(x_test[i] - x_test_prediction[i]))
        
        #error = np.sum(x_test[i] - x_test_prediction[i])
        test_error.append(error)

    for j in np.arange(len(x_fault)):
        error = np.sum(np.square(x_fault[j] - x_fault_prediction[j]))
        #error = np.sum(x_fault[j] - x_fault_prediction[j])

        fault_error.append(error)
    
    x = np.arange(len(x_test))
    
    plt.plot(x, test_error)
    plt.plot(x, fault_error)
    plt.show()

    print("Test_maximum_error:{}====Test_minimum_error:{}".format(np.max(test_error), np.min(test_error)))
    print("Fault_maximum_error:{}====Fault_minimum_error:{}".format(np.max(fault_error), np.min(fault_error)))