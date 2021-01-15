"""plots for visualization"""
from matplotlib import pyplot as plt


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





