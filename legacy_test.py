'''
 # @ Author: Hongi
 # @ Create Time: 2020-06-24 17:16:55
 # @ Modified by: Your name
 # @ Modified time: 2020-06-24 17:17:02
 # @ Description: Test and compute reconstruction error for each sample
 '''

import os
import numpy as np
import tensorflow as tf
from autoencoder.net import MLPAutoEncoder
from autoencoder import dataset
from autoencoder import config as cfg
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Build Model 
    model = MLPAutoEncoder()
        
    # Load pre-trained weights 
    weights_dir =  f"./checkpoints/{cfg.AUTOENCODER_WEIGHTS_DIR}"
    assert os.path.exists(weights_dir),\
    "The trained model not founded"  
    weights_path = weights_dir +"/cp.ckpt"
    model.load_weights(weights_path)

    # Load test-cases(normal and fault signal)
    x_test = dataset.data_normalization(np.loadtxt(cfg.TEST_DATASET_PATH))
    x_fault = dataset.data_normalization(np.loadtxt(cfg.TEST_FAULT_DATASET_PATH))


    if cfg.DATA_ADD_NOISE:
        x_test = dataset.add_noise(x_test)
        x_fault = dataset.add_noise(x_fault)

    # Convert to np.array for batch test
    test_pieces = np.array(x_test)
    fault_pieces = np.array(x_fault)

    # Do inference
    test_predictions = model.predict(test_pieces)
    fault_predictions = model.predict(fault_pieces)

    test_error = np.array([])
    fault_error = np.array([])

    # test_num == fault_num 
    test_num =len(test_pieces)
    fault_num = len(fault_pieces)
    num_covered_nodes_normal = []
    num_covered_nodes_fault = []

    # computing recostruction error and coverage rate for each sample
    for i in np.arange(test_num):
        distance_normal = np.square(x_test[i] - test_predictions[i])
        error = np.mean(distance_normal)
        covered_nodes_normal = np.where(distance_normal < cfg.TEST_THRESHOLD_COVERAGE)[0]

        test_error = np.append(test_error, error)
        num_covered_nodes_normal.append(len(covered_nodes_normal))

    for j in np.arange(fault_num):
        distance_fault = np.square(x_fault[j] - fault_predictions[j])
        error = np.mean(distance_fault)
        covered_nodes_fault = np.where(distance_fault < cfg.TEST_THRESHOLD_COVERAGE)[0]

        fault_error = np.append(fault_error,error)
        num_covered_nodes_fault.append(len(covered_nodes_fault))
        
    # Experiments for dual sensors

    i = 0
    j = 0
    k = 0

    for s in np.arange(test_num):

        if test_error[s] < cfg.TEST_THRESHOLD and fault_error[s] > cfg.TEST_THRESHOLD:
            i += 1
        
        if test_error[s] < fault_error[s]:
            j += 1
        
        if num_covered_nodes_normal[s] >= num_covered_nodes_fault[s]:
            k += 1
    
    sensor_novelty_detection_accuracy = i / test_num
    sensor_novelty_error_accuracy = j / test_num
    sensor_novelty_coverage_accuracy = k / test_num
    
    print(sensor_novelty_detection_accuracy)
    print(sensor_novelty_error_accuracy)
    print(sensor_novelty_coverage_accuracy)
    
    # To compute accuracy for detection
    # Range of reconstruction error
    error_max = np.max(np.append(test_error, fault_error))
    error_min = np.min(np.append(test_error, fault_error))
    
    # sampling 100 points from error interval to seek optimal threshold
    threshold = np.linspace(error_min, error_max, 100)
    coverage_range = np.arange(100)

    normal_test_accuracy = []
    abnormal_test_accuracy = []
    total_test_accuracy = []

    normal_test_coverage_accuracy = []
    abnormal_test_coverage_accuracy = []
    total_test_coverage_accuracy = []

    for t in threshold:

        normal_indexs = np.where(test_error <= t)
        num_normal_index = len(normal_indexs[0])
        normal_accuracy = num_normal_index / test_num
        normal_test_accuracy.append(normal_accuracy)

        fault_indexs = np.where(fault_error > t)
        num_fault_indexs = len(fault_indexs[0])
        abnormal_accuracy =num_fault_indexs / fault_num
        abnormal_test_accuracy.append(abnormal_accuracy)

        total_test_accuracy.append((normal_accuracy + abnormal_accuracy) / 2)

    for tc in coverage_range:

        normal_indexs = np.where(num_covered_nodes_normal > tc)[0]
        num_normal_index = len(normal_indexs)
        coverage_accuracy = num_normal_index / test_num
        normal_test_coverage_accuracy.append(coverage_accuracy)

        fault_indexs = np.where(num_covered_nodes_fault < tc)[0]
        num_fault_indexs = len(fault_indexs)
        coverage_abnormal_accuracy = num_fault_indexs / fault_num
        abnormal_test_coverage_accuracy.append(coverage_abnormal_accuracy)

        total_test_coverage_accuracy.append((coverage_accuracy + coverage_abnormal_accuracy) / 2)


    max_total_test_accuracy = np.max(total_test_accuracy)
    threshold_performance_index = np.where(total_test_accuracy == max_total_test_accuracy)[0]
    threshold_performance = threshold[threshold_performance_index]

    max_total_test_coverage_accuracy = np.max(total_test_coverage_accuracy)
    coverage_threshold_performance_index = np.where(total_test_coverage_accuracy == max_total_test_coverage_accuracy)
    coverage_threshold_performance = coverage_range[coverage_threshold_performance_index]

        
    # Visualization
    # Visualizing reconstruction error for each test_cases(normal vs. abnormal)
  
    # plt.subplot(2, 1, 1)
    # x = np.arange(len(x_test))
    # plt.plot(x, test_error, 'b', label='Error for normal signals')
    # plt.plot(x, fault_error, 'r', label='Error for abnormal signals')
    # plt.xlabel('Samples')
    # plt.ylabel(' Mean Squared Error')
    # plt.legend(loc='upper right')

    # # Visualizing coverage rate for each test_cases(normal vs. abnormal)
    # plt.subplot(2, 1, 2)
    # plt.plot(x, num_covered_nodes_normal, 'b', label='Coverage rate for normal signals')
    # plt.plot(x, num_covered_nodes_fault, 'r', label='Coverage rate for abnormal signals')
    # plt.xlabel('Samples')
    # plt.ylabel('Coverage rate')
    # plt.legend(loc='upper right')


    # Visualizing detection accuracy
    plt.subplot(2 ,1 ,1)
    t = threshold
    plt.plot(t, normal_test_accuracy,'g', label='Accuracy for normal test cases')
    plt.plot(t, abnormal_test_accuracy, 'c', label='Accuracy for abnormal test cases')
    plt.plot(t, total_test_accuracy,'k', label='Accuracy for all cases')
    plt.plot(threshold_performance, max_total_test_accuracy, 'p', label='Accuracy:{:.2%}--Threshold:{:.4}'.format(max_total_test_accuracy, threshold_performance[0]))
    plt.xlabel('Threshold(error based)')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

    # Visualizing detection accuracy based on coverage
    plt.subplot(2 ,1 ,2)
    c = np.arange(100)
    plt.plot(c, normal_test_coverage_accuracy, 'g', label='Accuracy for normal test cases')
    plt.plot(c, abnormal_test_coverage_accuracy, 'c', label='Accuracy for abnormal test cases')
    plt.plot(c, total_test_coverage_accuracy,'k', label='Accuracy for all cases')
    plt.plot(coverage_threshold_performance[0], max_total_test_coverage_accuracy, 'p', label='Accuracy:{:.2%}--Threshold:{}'.format(max_total_test_coverage_accuracy, coverage_threshold_performance[0]))
    plt.xlabel('Threshold(coverage rate based)')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Visualize detection accurracy in dual sensors architecture
    plt.show()

