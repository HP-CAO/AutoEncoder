"""searching for counter examples little by little"""

import numpy as np
from autoencoder import config as cfg
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder.hard_search import hard_search


if __name__ == "__main__":

    # build model & load pretrained weights
    model = MLPAutoEncoder().autoencoder
    weights_dir = cfg.AUTOENCODER_WEIGHTS_DIR
    model.load_weights(weights_dir)

    test_dataset = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))

    if cfg.DATA_ADD_NOISE:
        test_dataset = dataset.add_noise(test_dataset)

    sample_index = 1

    # traversing all samples in test dataset
    for sample in test_dataset:
        hard_search(sample, model, sample_index)
        sample_index += 1

    print("Searching completed")
