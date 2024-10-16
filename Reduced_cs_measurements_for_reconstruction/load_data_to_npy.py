# Will store my data as a single np array
# each element of the np array will be a tuple- (power, label)
# will store this data in a npy file

import os
import numpy as np
import matplotlib.pyplot as plt


def load_data_with_labels(main_folder_path):
    data_list = []

    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith('CS_20x20'):
            file_path = os.path.join(folder_path, 'data', 'measurements.npy')

            if os.path.exists(file_path):
                data = np.load(file_path)
                data_list.append(data[:,0])
            else:
                print(f"File not found: {file_path}")
        else:
            print(f"Not a directory: {folder_path}")

    return np.array(data_list)

# load data from the first pattern
data1 = load_data_with_labels(r"C:\Users\DMD Experiment\Desktop\arman\Code ver 7\Runs\2024_05_24\NP_133506")

label = np.zeros(len(data1))
data_set_0 = list(zip(data1, label))

# load data from the 2nd pattern
data2 = load_data_with_labels(r"C:\Users\DMD Experiment\Desktop\arman\Code ver 7\Runs\2024_05_23\NP_174118")
label = np.ones(len(data2))

data_set_1 = list(zip(data2, label))
# print((data_set_1[0]))

# dataset fit both the labels
combined = data_set_0 + data_set_1
data_set = np.array(combined, dtype=object)

# print(len(data_set))

np.random.shuffle(data_set)

np.save('data_set.npy', data_set)
