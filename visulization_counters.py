'''
 # @ Author: Hongi
 # @ Create Time: 2020-07-27 18:05:18
 # @ Modified by: Your name
 # @ Modified time: 2020-07-27 18:05:23
 # @ Description: Visulizing counter examples
 '''
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(100)
counters_dir = "39counter_examples.txt"
signalbase_dir = "39signal_base.txt"

counter_examples = np.loadtxt(counters_dir)
signalbase = np.loadtxt(signalbase_dir)

for counter in counter_examples:
    plt.subplot(2, 1, 1)
    plt.plot(x, counter)

plt.subplot(2, 1, 2)
plt.plot(x, signalbase)     
    
plt.show()