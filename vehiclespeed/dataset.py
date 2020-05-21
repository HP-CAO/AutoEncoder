'''
 # @ Author: Hongi
 # @ Create Time: 2020-05-20 16:41:00
 # @ Modified by: Your name
 # @ Modified time: 2020-05-20 16:43:03
 # @ Description: Create vehiclespeed signal dataset
 '''

import numpy as np



def train_dataset():

    train_dataset=[] 
    for i in range(8):
        train_data = np.loadtxt("./vehiclespeed/{}.txt".format(i+1))
        train_data = train_data[:,1]
        j = 0
        while True:   
            train_dataset.append(train_data[j: j + 50])
            j += 10
            if j + 50 > len(train_data):
                break
    np.random.shuffle(np.array(train_dataset))       
    
    return train_dataset

def test_dataset():

    test_dataset=[]
    test_data = np.loadtxt("./vehiclespeed/9.txt")
    test_data = test_data[:,1] 
    j = 0

    while True:
        test_dataset.append(test_data[j: j + 50])
        j += 10
        if j + 50 > len(test_data):
            break
    
    np.random.shuffle(np.array(test_dataset))
    
    return test_dataset

def spike_fault():
    
    spike_fault=[]
    fault_data = np.loadtxt("./vehiclespeed/10.txt")
    fault_data = fault_data[:, 1]
    j = 0

    while True:
        
        signal_slice = fault_data[j: j + 50]
        ''' create spike point based on normal signal: x(t) = x(t) + vδ(t),  '''
        ''' randomly pick one of the point from signal slice'''
        ''' randomly set amplititude v --| x = x(t)*(1+v) where v[0,1] '''

        v = np.random.random_sample()  ## create a random number [0, 1)
        spike_index = np.random.choice(len(signal_slice))
        signal_slice[spike_index] = signal_slice[spike_index] * (1 + v)     
        spike_fault.append(signal_slice)
        j += 10
        
        if j + 50 > len(fault_data):
            break
    
    np.random.shuffle(np.array(spike_fault))
    
    return spike_fault

def save_dataset(filename, dataset):
    """save dataset to .txt file"""

    with open(filename, mode='w', encoding='utf-8')as f:
        np.savetxt(filename, dataset)
    print(f"{filename} saved successfully")

if __name__ == "__main__":
    

    train_dataset = train_dataset()
    test_dataset = test_dataset()
    fault_dataset = spike_fault()
 
    save_dataset('speed_train', train_dataset)
    save_dataset('speed_test',test_dataset)
    save_dataset('speed_fault', fault_dataset)
