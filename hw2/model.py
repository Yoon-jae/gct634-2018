# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
#
# Jongpil Lee
#

from __future__ import print_function
import torch
import torch.nn as nn

# batch_size = 64
# num_frames = 512

# model class
class model_1DCNN(nn.Module):
    def __init__(self):
        super(model_1DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
        )

        self.fc = nn.Sequential(
            nn.Linear(96, 32),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        # input x: minibatch x 128 x 256

        out = self.conv0(x)  # 5 32 63
        out = self.conv1(out)  # 5 32 7

        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc(out)

        # out = nn.functional.log_softmax(out)
        # out = self.activation(out)

        return out


# model class
class model_2DCNN(nn.Module):
    def __init__(self):
        super(model_2DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=8),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Dropout(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(420, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # input x: minibatch x 128 x 256

        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # batch_size, 1, 128, 256

        out = self.conv0(x)  # batch_size, 10, 31, 63
        # print(out.shape)
        out = self.conv1(out)  # batch_size, 20, 3, 7
        # print(out.shape)
        # exit()
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2) * out.size(3))
        out = self.fc(out)

        return out
