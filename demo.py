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

from main import AutoEncoder


def data_normalization(data_set):
    """Do gloabal normalization for data"""
    # (xi-Xmin)/(Xmax-min)
    
    data_set = (data_set-np.min(data_set))/(np.max(data_set)-np.min(data_set))

    return data_set

    
def load_demo_data():

    """load demo data from speed_test.txt and spike_fault.txt for test"""
    ## Data_dir checking  
    try:
        os.path.exists('speed_train.txt')
        os.path.exists('speed_test.txt')
        os.path.exists('spike_fault.txt')
    except FileNotFoundError:
        print("Have you ever created test/fault dataset?")
    
    ## load_ testing data
    test_data = np.loadtxt('speed_test.txt')
    x_test = test_data
    ## laod_fault data
    fault_data = np.loadtxt('spike_fault.txt')
    x_fault = fault_data
    
    print("----------Data loaded!----------")
    return x_test, x_fault

def data_pick(test_ds, fault_ds, n=2):
    """Randomly pick several pieces of data for demo test"""
    
    test_indexes = np.random.choice(len(test_ds), n)
    test_pieces = []
    for i in test_indexes:
        test_pieces.append(test_ds[i])
    
    fault_indexes = np.random.choice(len(fault_ds), n)
    fault_pieces = []
    for i in fault_indexes:        
        fault_pieces.append(fault_ds[i])
    
    ## data normalization
    test_pieces = data_normalization(test_pieces)
    fault_pieces = data_normalization(fault_pieces)

    #print(test_pieces)
    return test_pieces, fault_pieces


if __name__ == "__main__":
    

    training_time_stamp = '20200624-192943'
    weights_dir =  f"./training_track/{training_time_stamp}"
        
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  

    weights_path = weights_dir +"/cp.ckpt"
    
    model = AutoEncoder()
    model.load_weights(weights_path)

    x_test, x_fault = load_demo_data()
    test_pieces, fault_pieces = data_pick(x_test, x_fault,5) ## Pick 5 samples from test_data and fault_data each
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

    
