import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN


class RNN(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                hidden_size=16,
                bidirectional=False
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.rnn = nn.RNN(input_size=input_channels, hidden_size=hidden_size,
                        num_layers=3,
                        batch_first=True,
                        bidirectional=bidirectional
                        )
        self.fc = nn.Linear(input_len * hidden_size, output_len * output_channels)

    def forward(self, x): # x: (batch_size, input_len, input_channels)
        x, hn = self.rnn(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, hidden_size)
        x = x.contiguous() # convert to contiguous memory
        x = x.view(x.size(0), -1) # (batch_size, input_len, hidden_size) -> (batch_size, input_len * hidden_size)
        x = self.fc(x) # (batch_size, input_len * hidden_size) -> (batch_size, output_len * output_channels)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len * output_channels) -> (batch_size, output_len, output_channels)
        return x


class LSTM(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                hidden_size=16,
                bidirectional=False
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=bidirectional
                            )
        self.fc = nn.Linear(input_len*hidden_size, output_len*output_channels)

    def forward(self, x): # x: (batch_size, input_len, input_channels)
        x, (hn, cn) = self.lstm(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, hidden_size)
        x = x.contiguous() # convert to contiguous memory
        x = x.view(x.size(0), -1) # (batch_size, input_len, hidden_size) -> (batch_size, input_len*hidden_size)
        x = self.fc(x) # (batch_size, input_len*hidden_size) -> (batch_size, output_len*output_channels)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x


class GRU(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                hidden_size=16,
                bidirectional=False
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.gru = nn.GRU(input_size=input_channels, hidden_size=hidden_size,
                        num_layers=3,
                        batch_first=True,
                        bidirectional=bidirectional
                        )
        self.fc = nn.Linear(input_len*hidden_size, output_len*output_channels)

    def forward(self, x): # x: (batch_size, input_len, input_channels)
        x, hn = self.gru(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, hidden_size)
        x = x.contiguous() # convert to contiguous memory
        x = x.view(x.size(0), -1) # (batch_size, input_len, hidden_size) -> (batch_size, input_len*hidden_size)
        x = self.fc(x) # (batch_size, input_len*hidden_size) -> (batch_size, output_len*output_channels)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x