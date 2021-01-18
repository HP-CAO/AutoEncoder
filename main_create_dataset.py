from data.create_dateset import create_dataset, save_dataset, create_spike_fault

if __name__ == "__main__":
    train_dataset, test_dataset = create_dataset()
    save_dataset('./data/speed_train.txt', train_dataset)
    save_dataset('./data/speed_test.txt', test_dataset)
    spike_fault = create_spike_fault("./data/speed_test.txt")
    save_dataset('./data/speed_fault.txt', spike_fault)
