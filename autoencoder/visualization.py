"""plots for visualization"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors


def ori_new_signals_plot(n_dim, signal_base, signal_new):
    
    """
    To compare original signals and new signals
    signal_base: The original signal
    signal_new: Signals generated after searching  
    n_dim: The dimensions of signal 
    
    """
    t = range(n_dim)
    plt.subplot(1, 1, 1)
    plt.plot(t, signal_base, label='signal_base')  
    plt.plot(t, signal_new, label='signal_new')
    plt.legend(loc='upper right')
    plt.show()


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


def plot_experiment_result(x, y, z):
    """
    plot result of the experiments
    """

    plt.scatter(x, y, c="b", alpha=1.0, marker="*",
                label="counters found (*E+03)")
    plt.plot(x, y, c='b', alpha=0.3)
    plt.scatter(x, z, c="g", alpha=0.5, marker="v",
                label="F1_score")
    plt.plot(x, z, c='g', alpha=0.3)
    plt.xlabel("Threshold (*E-05)")
    plt.ylabel("Y")
    plt.legend(loc='upper left')
    plt.show()
