'''
 # @ Author: Your Hongi
 # @ Create Time: 2020-07-20 13:17:53
 # @ Modified by: Your name
 # @ Modified time: 2020-07-20 13:18:04
 # @ Description: Searching for counter examples little by little.
 '''

import os 
import numpy as np

from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors



def hard_search(sample):
    all_error_list = [] # collection of error_lists

    for i in range(cfg.DATA_INPUT_DIMENSION):
        
        error_list = []  # errors for each step searching on ith time step 
        spike_faults = []
        counter_examples = []
        # create spike fault based on selected sample
        # searching steps = 1000
        for j in range(1000):
            
            signal = np.copy(sample)
            signal[i] = signal[i] + cfg.DATA_SPIKE_FAULT_MIN_VALUE * (1 + 0.009 * j) # searching rate =0.009
            spike_faults.append(signal)

        spike_faults = np.array(spike_faults)
        predictions = model.predict(spike_faults)
        
        for m in range(len(spike_faults)):    
            error = np.mean(np.square(spike_faults[m]-predictions[m]))
            error_list.append(error)

        if error_list[0] == np.min(error_list):
            print('[Safe]: Searching {} th sample, at {} th time step, No counter examples found'.format(sample_index, i + 1))
             
        else:
            index = np.where(error_list < error_list[0])
            counter_examples = [spike_faults [n] for n in index]
            print('[Unsafe]: >>>>>>>>>> At {} th time step, Found {} counter examples'.format(i,len(index)))        
            vis_step_error(error_list)   
            signalbase_file = open("{}signal_base.txt".format(sample_index), "w")
            np.savetxt(signalbase_file, sample)
            counters_file = open("{}counter_examples.txt".format(sample_index), "w")
            for row in counter_examples:
                np.savetxt(counters_file, row)
        all_error_list.append(error_list)
        vis_step_error(error_list)
    #vis_all_error_3d(all_error_list)

def vis_all_error_3d(all_error_list):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    all_error_list = np.array(all_error_list)
    pair = []
    
    xs = np.arange(1000)
    zs = np.arange(100)

    for z in zs:
        ys = all_error_list[z]
        pair.append(list(zip(xs, ys)))
    
    poly = LineCollection(pair)
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X: searching step')
    ax.set_xlim3d(0, 1000)
    ax.set_ylabel('Y: time step')
    ax.set_ylim3d(0, 100)
    ax.set_zlabel('Z: reconstruction error')
    ax.set_zlim3d(0, 0.005)
    plt.show()
 

def vis_step_error(error_list):
    x = range(len(error_list))
    y = error_list

    plt.plot(x, y)
    plt.xlabel('X: searching steps')
    plt.ylabel('Y: reconstruction error')
    plt.show()
    
if __name__ == "__main__":

    # build model & load pretrained weights
    model = MLPAutoEncoder()
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    test_dataset = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))

    if cfg.DATA_ADD_NOISE:
        test_dataset = dataset.add_noise(test_dataset)

    sample_index = 1

    # traversing all samples in test dataset
    for sample in test_dataset:        
        hard_search(sample)
        sample_index += 1
        
    print("Searching completed")


       
        
            
