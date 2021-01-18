"""
creat speed signal dataset train/test
"""

import numpy as np
from autoencoder import config as cfg


def create_dataset():

    dataset = []
    for i in range(10):
        data = np.loadtxt("./data/signal_data/{}.txt".format(i + 1))
        data = data[:, 1]
        j = 0

        while True:   
            dataset.append(data[j: j + 100])
            j += 10
            if j + 100 > len(data):
                break
    
    np.random.shuffle(dataset)

    n = len(dataset)
    # max_value = np.max(dataset)
    # min_value = np.min(dataset)
    #
    # print(max)
    # print("-----")
    # print(min)

    # spilt dataset into train/test subset (0.85:0.15)

    dataset_train = dataset[0:round(0.85*n)].copy()
    dataset_test = dataset[round(0.85*n): n].copy()

    return dataset_train, dataset_test


def create_spike_fault(path):

    """
    create spike signal based on normal signal: x(t) = x(t) + vδ(t),
    randomly pick one of the point from signal slice
    randomly set amplitude v --| x = x(t)*(1+v) where v[0,1]
    """
    
    signal_set = np.loadtxt(path)

    for i in range(len(signal_set)):
        v = np.random.random_sample()  # create a random number [0, 1)
        spike_index = np.random.choice(len(signal_set[i]))
        # create spike fault according to minimum height of spike fault
        #signal_set[i][spike_index] = signal_set[i][spike_index] + cfg.DATA_SPIKE_FAULT_MIN_VALUE
        signal_set[i][spike_index] = signal_set[i][spike_index] * (1 + v)

    return signal_set


def save_dataset(filename, dataset):
    """save dataset to .txt file"""

    with open(filename, mode='w', encoding='utf-8')as f:
        np.savetxt(filename, dataset)
    f.close()
    print(f"{filename} saved successfully")


