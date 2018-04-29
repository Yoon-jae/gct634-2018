# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import numpy as np


def load_data(label_path, mel_path, melBins, frames):
    # read train / valid / test lists
    y_train_dict = {}
    y_valid_dict = {}
    y_test_dict = {}

    with open(label_path + 'train_filtered.txt') as f:
        train_list = f.read().splitlines()
        for line in train_list:
            y_train_dict[line] = line.split('/')[0]
    with open(label_path + 'valid_filtered.txt') as f:
        valid_list = f.read().splitlines()
        for line in valid_list:
            y_valid_dict[line] = line.split('/')[0]
    with open(label_path + 'test_filtered.txt') as f:
        test_list = f.read().splitlines()
        for line in test_list:
            y_test_dict[line] = line.split('/')[0]

    # labels
    genres = list(set(y_train_dict.values()))
    print(genres)
    for iter in range(len(y_train_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_train_dict[train_list[iter]]:
                y_train_dict[train_list[iter]] = iter2
    for iter in range(len(y_valid_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_valid_dict[valid_list[iter]]:
                y_valid_dict[valid_list[iter]] = iter2
    for iter in range(len(y_test_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_test_dict[test_list[iter]]:
                y_test_dict[test_list[iter]] = iter2

    # load data
    x_train = np.zeros((len(train_list), melBins, frames))
    y_train = np.zeros((len(train_list),))
    for iter in range(len(train_list)):
        x_train[iter] = np.load(mel_path + train_list[iter].replace('.wav', '.npy'))
        y_train[iter] = y_train_dict[train_list[iter]]

    x_valid = np.zeros((len(valid_list), melBins, frames))
    y_valid = np.zeros((len(valid_list),))
    for iter in range(len(valid_list)):
        x_valid[iter] = np.load(mel_path + valid_list[iter].replace('.wav', '.npy'))
        y_valid[iter] = y_valid_dict[valid_list[iter]]

    x_test = np.zeros((len(test_list), melBins, frames))
    y_test = np.zeros((len(test_list),))
    for iter in range(len(test_list)):
        x_test[iter] = np.load(mel_path + test_list[iter].replace('.wav', '.npy'))
        y_test[iter] = y_test_dict[test_list[iter]]

    # normalize the mel spectrograms
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train -= mean
    x_train /= std
    x_valid -= mean
    x_valid /= std
    x_test -= mean
    x_test /= std

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def augment_data(X, y, frame_size=256, hop_size=128, melBins=128):
    num_of_examples, num_of_features, total_frame = X.shape  # 443 128 1287
    starts = [start for start in range(0, total_frame - frame_size, hop_size)]
    split_size = len(starts)

    aug_x = np.zeros((split_size * num_of_examples, melBins, frame_size))
    aug_y = np.zeros((split_size * num_of_examples,))

    pointer = 0
    for sample_index, sample in enumerate(X):
        for window_index, start in enumerate(starts):
            aug_x[pointer + window_index] = sample[:, start:start + frame_size]
            aug_y[pointer + window_index] = y[sample_index]
        pointer += split_size

    # print("aug_x :", aug_x.shape)
    # print("aug_y :", aug_y.shape)

    return aug_x, aug_y


if __name__ == '__main__':
    label_path = './gtzan/'
    mel_path = './gtzan_mel/'
    melBins = 128
    hop = 512
    frames = int(29.9 * 22050.0 / hop)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(label_path, mel_path, melBins, frames)
    augment_data(x_train, y_train)
