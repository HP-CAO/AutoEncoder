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


def hard_search(sample):

    for i in range(cfg.DATA_INPUT_DIMENSION):
        
        error_list = []
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
            
            signalbase_file = open("{}signal_base.txt".format(sample_index), "w")
            np.savetxt(signalbase_file, sample)
            counters_file = open("{}counter_examples.txt".format(sample_index), "w")
            for row in counter_examples:
                np.savetxt(counters_file, row)

        

if __name__ == "__main__":

    # build model & load pretrained weights
    model = MLPAutoEncoder()
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    # sample a single signal for experiments
    # sample_index = np.random.randint(0, 300)
    # sample = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))[sample_index]
    # sample = dataset.add_noise(sample)

    test_dataset = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))

    if cfg.DATA_ADD_NOISE:
        test_dataset = dataset.add_noise(test_dataset)

    sample_index = 1

    for sample in test_dataset:        
        hard_search(sample)
        sample_index += 1
        
    
    print("Searching completed")


       
        
            
