import os
import numpy as np

from matplotlib import pyplot as plt
from autoencoder import config as cfg
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset

if __name__ == "__main__":

    # Build Model 
    model = MLPAutoEncoder().autoencoder

    # Load pre-trained weights 
    weights_dir = cfg.AUTOENCODER_WEIGHTS_DIR
    model.load_weights(weights_dir)
    test_dataset = np.loadtxt(cfg.TEST_DATASET_PATH)
    fault_dataset = np.loadtxt(cfg.TEST_FAULT_DATASET_PATH)

    if cfg.DATA_ADD_NOISE:
        test_dataset = dataset.add_noise(test_dataset)
        fault_dataset = dataset.add_noise(fault_dataset)

    # Load test-cases(normal and fault signal)
    x_test = dataset.data_normalization(test_dataset)
    x_fault = dataset.data_normalization(fault_dataset)

    # Randomly sample from dataset
    test_pieces, fault_pieces = dataset.data_pick(x_test, x_fault, cfg.TEST_SAMPLE_NUM)

    # Do inference
    test_predictions = model.predict(np.array(test_pieces))
    fault_predictions = model.predict(np.array(fault_pieces))

    com = np.array(test_predictions) == np.array(fault_predictions)
    result = com.all()

    # visualization
    x = np.arange(100)

    for i in range(cfg.TEST_SAMPLE_NUM):
        # subplot_1 for normal test cases
        plt.subplot(3, 1, 1)
        plt.plot(x, test_pieces[i], label='Normal signal')
        plt.plot(x, test_predictions[i], label='Reconstructed normal signal')
        plt.legend(loc='lower right')
        plt.title('Normal signal')

        # subplot_2 for abnormal test cases
        plt.subplot(3, 1, 2)
        plt.plot(x, fault_pieces[i], label='Abnormal signal')
        plt.plot(x, fault_predictions[i], label='Reconstructed abnormal signal')
        plt.legend(loc='lower right')
        plt.ylabel('magnitude:x(t)')
        plt.title('Abnormal signal')

        # subplot_3 for comparation between normal/abnormal reconstruction
        plt.subplot(3, 1, 3)
        plt.plot(x, test_predictions[i], label='Normal signal constructed')
        plt.plot(x, fault_predictions[i], label='Abnormal signal constructed')
        plt.legend(loc='lower right')
        plt.xlabel('time:t')
        plt.title('Reconstruction Comparation')

        plt.show()
