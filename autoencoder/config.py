'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:03:31
 # @ Modified by: Your name
 # @ Modified time: 2020-06-28 15:21:35
 # @ Description: Configuraion file/hyperparameters
 '''


# Weights options
AUTOENCODER_WEIGHTS_DIR = '20200805-135413'

# Data options 
DATA_NORMALIZATION_MAX = 135.105346243647
DATA_NORMALIZATION_MIN = -0.0769845108208287
DATA_INPUT_DIMENSION = 100
DATA_ADD_NOISE = True
DATA_NOISE_MEAN = 0
DATA_NOISE_DEVIATION = 0.003
DATA_SPIKE_FAULT_MIN_VALUE = 5 * DATA_NOISE_DEVIATION
DATA_SPIKE_FAULT_MIN_HEIGHT_RATIO = 1.1
DATA_SPIKE_FAULT_MAX_HEIGHT_RATIO = 2.0 

# Train options
TRAIN_EPOCHS = 1000
TRAIN_BATCH_SIZE = 64
TRAIN_DATASET_PATH = './data/speed_train.txt' 

# Test options
TEST_THRESHOLD = 0.000007605
TEST_THRESHOLD_COVERAGE = 0.000007605
TEST_DATASET_PATH = './data/speed_test.txt'
TEST_FAULT_DATASET_PATH = './data/speed_fault.txt'
TEST_SAMPLE_NUM = 1

# Optimization options
OPT_LEARNING_RATE = 0.1
