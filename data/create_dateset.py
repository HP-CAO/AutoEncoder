'''
 # @ Author: Hongi
 # @ Create Time: 2020-05-20 16:41:00
 # @ Modified by: Your name
 # @ Modified time: 2020-05-20 16:43:03
 # @ Description: Creating vehiclespeed signal dataset (train/test/fault)
 '''

import numpy as np


def create_dataset():

    dataset=[] 
    for i in range(10):
        data = np.loadtxt("./data/{}.txt".format(i + 1))
        data = data[:,1]
        j = 0

        while True:   
            dataset.append(data[j: j + 100])
            j += 10
            if j + 100 > len(data):
                break
    
    np.random.shuffle(dataset)

    ## Divide into train/test/fault dataset
          
    n = len(dataset)
    max=np.max(dataset)
    min=np.min(dataset)

    print(max)
    print("-----")
    print(min)
    
    train_dataset = dataset[0:round(0.85*n)].copy()
    test_dataset = dataset[round(0.85*n): n].copy()

    return train_dataset, test_dataset

def create_spike_fault():
    
    ''' create spike point based on normal signal: x(t) = x(t) + vδ(t),  '''
    ''' randomly pick one of the point from signal slice'''
    ''' randomly set amplititude v --| x = x(t)*(1+v) where v[0,1] '''
    
    signal_set = np.loadtxt("./data/speed_test.txt")

    for i in range(len(signal_set)):
        v = np.random.random_sample()  ## create a random number [0, 1)
        spike_index = np.random.choice(len(signal_set[i]))
        signal_set[i][spike_index] = signal_set[i][spike_index] * (1 + v/5 + 0.1) # we assume the minimum height ratio of spike fault is 0.1

    return signal_set


def save_dataset(filename, dataset):
    """save dataset to .txt file"""

    with open(filename, mode='w', encoding='utf-8')as f:
        np.savetxt(filename, dataset)
    print(f"{filename} saved successfully")

if __name__ == "__main__":
    

    train_dataset, test_dataset = create_dataset()
    save_dataset('./data/speed_train.txt', train_dataset)
    save_dataset('./data/speed_test.txt',test_dataset)
    spike_fault = create_spike_fault()
    save_dataset('./data/speed_fault.txt', spike_fault)