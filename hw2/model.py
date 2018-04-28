# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
#
# Jongpil Lee
#

from __future__ import print_function
import torch
import torch.nn as nn

batch_size = 100

# model class
class model_1DCNN(nn.Module):
    def __init__(self):
        super(model_1DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

        self.max_pool = nn.MaxPool1d(4, stride=4)
        self.avg_pool = nn.AvgPool1d(4, stride=4)

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, 10)
        )

        # self.activation = nn.Softmax()

    def forward(self, x):
        # input x: minibatch x 128 x 128
        out = self.conv0(x)  # 5 256 127
        out = self.conv1(out)  # 5 256 63
        out = self.conv2(out)  # 5 512 31

        out1 = self.max_pool(out)  # 512 7
        out2 = self.avg_pool(out)  # 512 7

        out = torch.cat([out1, out2], dim=2)
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc(out)

        out = nn.functional.log_softmax(out)
        # out = self.activation(out)

        return out