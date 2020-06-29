'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:04:13
 # @ Modified by: 
 # @ Modified time: 2020-06-28 14:04:44
 # @ Description: Data prepration/preprocessing
 '''
import autoencoder.config as cfg
import os
import numpy as np
import tensorflow as tf



def compute_norm_factors():
    dataset= np.loadtxt(cfg.TRAIN_DATASET_PATH, dtype=float)
    max = np.max(dataset)
    min = np.min(dataset)
    return print('max for train dataset:{}, --- min for train dataset: {}'.format(max, min))

def data_normalization(dataset):
    """Do gloabal normalization for data"""
    # (xi-Xmin)/(Xmax-min) 

    dataset = (dataset-cfg.DATA_NORMALIZATION_MIN)/(cfg.DATA_NORMALIZATION_MAX-cfg.DATA_NORMALIZATION_MIN)
    return dataset

def load_data(data_path):
    # load data for batch training/batch testing
    # Data_dir checking  
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


def data_pick(test_ds, fault_ds, n):
    """Randomly pick several pieces of data for demo test """

    indexes = np.random.choice(len(test_ds), n)
    test_pieces = []
    fault_pieces = []

    for i in indexes:
        test_pieces.append(test_ds[i])
        fault_pieces.append(fault_ds[i])

    return test_pieces, fault_pieces


