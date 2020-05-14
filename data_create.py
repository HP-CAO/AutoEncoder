import math
import numpy as np
import os
from matplotlib import pyplot as plt


def sin_signal_create(a, b, c, d):
    """ y= Asin(Bx+C)+D"""
    y = []
    a = 10*a
    b = 10*b*math.pi
    c = 10*c
    for x in np.arange(50): 
        y.append(a * math.sin(b*x + c) + d)
    
    # x=np.arange(50)
    # plt.plot(x,y) 
    # plt.show() 
    return y

def constant_fault_create(const):
    """ f = c"""
    const = const*10
    y = np.full(50,const)
    return y

def sin_signal_dataset():
    A = np.random.rand(10)  ## randomly create a list whose element is within [0,1)
    B = np.random.rand(10)
    C = np.random.rand(10)
    D = np.random.rand(10)
    Y = []
    for a in A:
        for b in B:
            for c in C:
                for d in D:
                    Y.append(sin_signal_create(a,b,c,d))

    n = len(Y)
    Y=np.array(Y)
    np.random.shuffle(Y)
    return Y[0:round(0.8*n)], Y[round(0.8*n):n]

def constant_fault_dataset():
    const = np.random.rand(1000)
    F = []
    for i in const:
        F.append(constant_fault_create(i))
    return F

def add_noise():
    """Add gaussian noise"""
    pass

def save_dataset(filename, Y):
    """ Save dataset to .txt file"""
    
    with open(filename, mode='w',encoding ='utf-8') as f:
        np.savetxt(filename, Y)
    print( f"{filename} saved successfully!")


if __name__ == "__main__":
    train, test = sin_signal_dataset()
    fault = constant_fault_dataset()
    train_dataset = 'train.txt'
    test_dataset = 'test.txt'
    fault_filename = 'fault.txt'
    save_dataset(train_dataset, train)
    save_dataset(test_dataset, test)
    save_dataset(fault_filename, fault)




