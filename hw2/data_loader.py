# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import random


# data loader

class gtzandata(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):

        # 30 sec -> 1287 frames
        # 6 sec  -> 256 frames
        mel = self.x[index]  # 128, 256

        # 6 sec -> 256 frames
        # num_frames = 256
        # start = random.randint(0, self.x[index].shape[1] - num_frames)
        # mel = self.x[index][:, start:start + num_frames]  # 128, 256

        entry = {'mel': mel, 'label': self.y[index]}

        return entry

    def __len__(self):
        return self.x.shape[0]