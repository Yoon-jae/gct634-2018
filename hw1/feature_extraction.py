# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
# Juhan Nam
#
# Apr-06-2018: developed version
# Yoonjae Cho

import os
import librosa
import numpy as np

data_path = './dataset/'
mfcc_path = './mfcc/'

# Feature parameter

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
WIN_LENGTH = 1024
N_MELS = 40
MFCC_DIM = 2


def get_matrix_feature(features):
    matrix = features[0]
    for feature in features[1:]:
        matrix = np.row_stack((matrix, feature))
    return matrix


def extract_feature(dataset):
    f = open(data_path + dataset + '_list.txt', 'r')

    for index, file_name in enumerate(f):
        # progress check
        if not (index % 100):
            print(dataset + ': ' + str(index) + ', completed')

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        # print file_path
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)


        # MFCC
        # STFT
        S = librosa.core.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

        # power spectrum
        D = np.abs(S) ** 2

        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, n_fft=N_FFT, n_mels=N_MELS)
        mel_S = np.dot(mel_basis, D)

        # log compression
        log_mel_S = librosa.power_to_db(mel_S)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)
        rmse = librosa.feature.rmse(y, frame_length=1024, hop_length=512)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=512)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y=y)), sr=sr)
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=2)

        feature = get_matrix_feature([
            zero_crossing_rate,
            rmse,
            spectral_centroid,
            spectral_bandwidth,
            chroma_cens,
            spectral_contrast,
            flux,
            mfcc
        ])

        # save feature as a file
        file_name = file_name.replace('.wav', '.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, feature)

    f.close()


if __name__ == '__main__':
    extract_feature(dataset='train')
    extract_feature(dataset='valid')
    extract_feature(dataset='test')
