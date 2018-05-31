'''
model_archive.py

A file that contains neural network models.
You can also make different model like CNN if you follow similar format like given RNN.
'''
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=64,
                            num_layers=1,
                            bidirectional=True,
                            dropout=0.25)
        self.linear = nn.Linear(2 * 64, num_classes)  # 2 for bidirection

        self.fc = nn.Sequential(
            nn.Linear(2 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x.shape : seq_length, batch_size, input_size -> RNN input

        output, hidden = self.lstm(x, None)
        # output.shape : seq_length, batch_size, output_size
        # hidden.shape : num_layer * direction, batch_size, hidden_size

        # output.shape : 2 8 128
        # output[-1].shape : 8 128
        output = self.fc(output[-1])

        return output


class CRNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=64,
                            num_layers=1,
                            bidirectional=True,
                            dropout=0.25)

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x.shape : seq_length, batch_size, input_size -> for RNN input
        # x.shape : 2, 8, 12

        x = x.view(x.size(1), x.size(2), x.size(0))
        # changed x : batch_size, input_size, seq_length
        # x.shape : 8, 12, 16

        output = self.conv(x)
        # 8, 32, 10
        # 4, 32, 6


        output = output.view(output.size(2), output.size(0), output.size(1))
        output, hidden = self.lstm(output, None)

        output = self.fc(output[-1])

        return output