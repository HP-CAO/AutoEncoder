"""Hard search for searching counter examples"""
import numpy as np

from autoencoder import config as cfg
from autoencoder.visualization import vis_step_error


def hard_search(selected_sample, model, sample_index):
    all_error_list = []  # collection of error_lists

    for i in range(cfg.DATA_INPUT_DIMENSION):

        error_list = []  # errors for each step searching on ith time step
        spike_faults = []
        counter_examples = []

        # create spike fault based on selected sample
        # searching steps = 1000

        for j in range(1000):
            signal = np.copy(selected_sample)
            signal[i] = signal[i] + cfg.DATA_SPIKE_FAULT_MIN_VALUE * (1 + 0.009 * j)  # searching rate =0.009
            spike_faults.append(signal)

        spike_faults = np.array(spike_faults)
        predictions = model(spike_faults)

        for m in range(len(spike_faults)):
            error = np.mean(np.square(spike_faults[m] - predictions[m]))
            error_list.append(error)

        if error_list[0] == np.min(error_list):
            print('[Safe]: Searching {} th sample, at {} th time step, No counter examples found'.format(sample_index,
                                                                                                         i + 1))
        else:
            index = np.where(error_list < error_list[0])
            counter_examples = [spike_faults[n] for n in index]
            print('[Unsafe]: >>>>>>>>>> At {} th time step, Found {} counter examples'.format(i, len(index)))
            vis_step_error(error_list)
            signalbase_file = open("{}signal_base.txt".format(sample_index), "w")
            np.savetxt(signalbase_file, selected_sample)
            counters_file = open("{}counter_examples.txt".format(sample_index), "w")
            for row in counter_examples:
                np.savetxt(counters_file, row)
        all_error_list.append(error_list)
        vis_step_error(error_list)



