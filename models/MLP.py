import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN

class MLP(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    A 3-layer MLP with ReLU activation, max pooling, and dropout.
    Many parameters
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                dropout=0.2,
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.fc1 = nn.Linear(input_len*input_channels, 256)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(32, output_channels*output_len)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x=x.view(x.size(0), self.input_len*self.input_channels) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x=self.fc1(x) # (batch_size, input_len*input_channels) -> (batch_size, 256)
        x=nn.functional.relu(x)
        x=self.pool1(x) # (batch_size, 256) -> (batch_size, 128)
        x=self.dropout1(x)
        x=self.fc2(x) # (batch_size, 128) -> (batch_size, 64)
        x=nn.functional.relu(x)
        x=self.pool2(x) # (batch_size, 64) -> (batch_size, 32)
        x=self.dropout2(x)
        x=self.fc3(x) # (batch_size, 32) -> (batch_size, output_channels*output_len)
        x = x.view(-1, self.output_len, self.output_channels) # (batch_size, output_channels*output_len) -> (batch_size, output_len, output_channels)
        return x