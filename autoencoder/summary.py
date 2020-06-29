'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-28 14:13:45
 # @ Modified by: Your name
 # @ Modified time: 2020-06-28 14:14:07
 # @ Description: Creating summaries for tensorboard
 '''

import tensorflow as tf
import random
import numpy as np
import os
import datetime

def build_summary_writer(current_time):

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/speed_train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/speed_test'
    fault_log_dir = 'logs/gradient_tape/' + current_time + '/spike_fault'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    fault_summary_writer = tf.summary.create_file_writer(fault_log_dir)

    return train_summary_writer, test_summary_writer, fault_summary_writer
