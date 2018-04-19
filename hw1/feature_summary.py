# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
# Juhan Nam
#
# Apr-06-2018: developed version
# Yoonjae Cho

import matplotlib.pyplot as plt
import numpy as np

data_path = './dataset/'
mfcc_path = './mfcc/'

FEATURE_DIM = 26


def mean_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(FEATURE_DIM, 1000))
    else:
        mfcc_mat = np.zeros(shape=(FEATURE_DIM, 200))

    for index, file_name in enumerate(f):
        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav', '.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mean pooling
        mfcc_mat[:, index] = np.mean(mfcc, axis=1)

    return mfcc_mat


if __name__ == '__main__':
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')
    test_data = mean_mfcc('test')

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
