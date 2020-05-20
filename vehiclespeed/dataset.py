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
        j=0
        while True:   
            train_dataset.append(train_data[j: j+50])
            j+=10
            if j+50 > len(train_data):
                break
    
    return train_dataset

def test_dataset():
    


def spike_fault():
    pass


train_dataset()