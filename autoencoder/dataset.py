"""dataset preparation and data preprocess"""

import autoencoder.config as cfg
import os
import numpy as np
import tensorflow as tf


def compute_norm_factors():
    """compute max/min value of the training dataset for Min-Max Normalization """
    dataset = np.loadtxt(cfg.TRAIN_DATASET_PATH, dtype=float)
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    return print('max for train dataset:{}, --- min for train dataset: {}'.format(max_value, min_value))


def data_normalization(dataset):
    """
    Max-Min normalization on dataset
    (xi-Xmin)/(Xmax-min)
    """
    dataset = (dataset-cfg.DATA_NORMALIZATION_MIN)/(cfg.DATA_NORMALIZATION_MAX-cfg.DATA_NORMALIZATION_MIN)
    return dataset


def add_noise(dataset):
    """"Adding Gaussian noise"""
    np.random.seed(2)
    noise = np.random.normal(cfg.DATA_NOISE_MEAN, cfg.DATA_NOISE_DEVIATION, cfg.DATA_INPUT_DIMENSION)
    dataset = np.add(dataset, noise)
    return dataset


def load_data(data_path):
    """
     load data for batch training/batch testing during training phrase
    """
    try:
        os.path.exists(data_path)
    except FileNotFoundError:
        print("Have you ever created train/tes/fault dataset?")
    
    data = data_normalization(np.loadtxt(data_path))
    
    if cfg.DATA_ADD_NOISE:
        data = add_noise(data)

    data = tf.cast(data, tf.float32)

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


