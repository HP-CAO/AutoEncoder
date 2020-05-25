'''
 # @ Author: Hongi
 # @ Create Time: 2020-05-20 16:41:00
 # @ Modified by: Your name
 # @ Modified time: 2020-05-20 16:43:03
 # @ Description: Create vehiclespeed signal dataset
 '''

import numpy as np



def create_dataset():

    dataset=[] 
    for i in range(10):
        
        data = np.loadtxt("./vehiclespeed/{}.txt".format(i + 1))
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
    train_dataset = dataset[0:round(0.7*n)].copy()
    test_dataset = dataset[round(0.7*n): round(0.85*n)].copy()
    fault_dataset = dataset[round(0.85*n): n].copy()

    return train_dataset, test_dataset, fault_dataset



def create_spike_fault():
    
    ''' create spike point based on normal signal: x(t) = x(t) + vδ(t),  '''
    ''' randomly pick one of the point from signal slice'''
    ''' randomly set amplititude v --| x = x(t)*(1+v) where v[0,1] '''
    
    signal_set = np.loadtxt("./speed_fault.txt")

    for i in range(len(signal_set)):
        v = np.random.random_sample()  ## create a random number [0, 1)
        spike_index = np.random.choice(len(signal_set[i]))
        signal_set[i][spike_index] = signal_set[i][spike_index] * (1 + v/4)

    return signal_set


def save_dataset(filename, dataset):
    """save dataset to .txt file"""

    with open(filename, mode='w', encoding='utf-8')as f:
        np.savetxt(filename, dataset)
    print(f"{filename} saved successfully")

if __name__ == "__main__":
    

    train_dataset, test_dataset, fault_dataset = create_dataset()
    save_dataset('speed_train.txt', train_dataset)
    save_dataset('speed_test.txt',test_dataset)
    save_dataset('speed_fault.txt', fault_dataset)

    spike_fault = create_spike_fault()

    save_dataset('spike_fault.txt', spike_fault)
