'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:03:31
 # @ Modified by: Your name
 # @ Modified time: 2020-06-28 15:21:35
 # @ Description: Configuraion file/hyperparameters
 '''


# Weights options
AUTOENCODER_WEIGHTS_DIR = '20200629-150656'

# Data options 
DATA_NORMALIZATION_MAX = 135.105346243647
DATA_NORMALIZATION_MIN = -0.0769845108208287


# Train options
TRAIN_EPOCHS = 500
TRAIN_BATCH_SIZE = 64
TRAIN_DATASET_PATH = './data/speed_train.txt' 

# Test options
TEST_THRESHOLD = None
TEST_DATASET_PATH = './data/speed_test.txt'
TEST_FAULT_DATASET_PATH = './data/speed_fault.txt'
TEST_SAMPLE_NUM = 5
