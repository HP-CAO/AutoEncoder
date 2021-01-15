"""
visualizing the training process using tensorboard
"""

import tensorflow as tf

import datetime


def build_summary_writer():

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/speed_train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/speed_test'
    fault_log_dir = 'logs/gradient_tape/' + current_time + '/spike_fault'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    fault_summary_writer = tf.summary.create_file_writer(fault_log_dir)

    return train_summary_writer, test_summary_writer, fault_summary_writer
