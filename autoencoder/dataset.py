'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:04:13
 # @ Modified by: 
 # @ Modified time: 2020-06-28 14:04:44
 # @ Description: Data prepration/preprocessing
 '''
from autoencoder import config as cfg
import os
import numpy as np
import tensorflow as tf

def data_normalization(dataset):
    """Do gloabal normalization for data"""
    # (xi-Xmin)/(Xmax-min)
    dataset = (dataset-np.min(dataset))/(np.max(dataset)-np.min(dataset))
    return dataset

def load_data(data_path):
    ## Data_dir checking  
    try:
        os.path.exists(data_path)
    except FileNotFoundError:
        print("Have you ever created train/tes/fault dataset?")
    
    ## Load data and normalization
    data = data_normalization(np.loadtxt(data_path))
    x_data = data
    y_data = data
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(cfg.TRAIN_BATCH_SIZE)

    print("----------Data loaded!----------")
    return dataset


def data_pick(test_ds, fault_ds):
    """Randomly pick several pieces of data for demo test"""
    
    test_indexes = np.random.choice(len(test_ds), cfg.TEST_SAMPLE_NUM)
    test_pieces = []
    for i in test_indexes:
        test_pieces.append(test_ds[i])
    

    fault_indexes = np.random.choice(len(fault_ds), cfg.TEST_SAMPLE_NUM)
    fault_pieces = []
    for i in fault_indexes:        
        fault_pieces.append(fault_ds[i])
    
    ## data normalization
    test_pieces = data_normalization(test_pieces)
    fault_pieces = data_normalization(fault_pieces)

    #print(test_pieces)
    return test_pieces, fault_pieces