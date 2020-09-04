'''
 # @ Author: Hongi
 # @ Create Time: 2020-07-27 18:05:18
 # @ Modified by: Your name
 # @ Modified time: 2020-07-27 18:05:23
 # @ Description: Visulizing counter examples
 '''
import numpy as np
from matplotlib import pyplot as plt
def vis_counters():
    x = np.arange(100)
    counters_dir = "2counter_examples.txt"
    signalbase_dir = "2signal_base.txt"

    counter_examples = np.loadtxt(counters_dir)
    signalbase = np.loadtxt(signalbase_dir)

    for counter in counter_examples:
        plt.subplot(2, 1, 1)
        plt.plot(x, counter)

    plt.subplot(2, 1, 2)
    plt.plot(x, signalbase)     
        
    plt.show()


def plot():
    # T1 = 6.375966677296674e-06     found 0
    # T2 = 6.579171963494446e-05     found 1888
    # T3 = 3.6083843156120565e-05    found 1564
    # T4 = 2.122990491670862e-05    found 64
    # T5 = 2.8656874036414592e-05    found 1027
    # T6 = 2.4943389476561606e-05    found521
    y = [0, 0.064, 0.521, 1.027, 1.564, 1.888]
    x = [0.637, 2.1229, 2.4943, 2.865, 3.6083, 6.5792]
    z = [0.847, 0.949, 0.962, 0.967, 0.642, 0.662]
    
    
    # x = np.arange(0.0, 50.0, 2.0)
    # y = x ** 1.3 + np.random.rand(*x.shape) * 30.0
    # s = np.random.rand(*x.shape) * 800 + 500

    plt.scatter(x, y, c="b", alpha=1.0, marker="*",
                label="counters found (*E+03)")
    plt.plot(x, y, c='b',alpha=0.3)
    plt.scatter(x, z, c="g", alpha=0.5, marker="v",
                label="F1_score")
    plt.plot(x,z, c='g',alpha=0.3)
    plt.xlabel("Threshold (*E-05)")
    plt.ylabel("Y")
    plt.legend(loc='upper left')
    plt.show()

plot()