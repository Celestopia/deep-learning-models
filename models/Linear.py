import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclass import TimeSeriesNN


class LLinear(TimeSeriesNN):
    """
    Just one Linear layer
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py; Modified afterwards.
    Note that input_channels == output_channels
    """
    def __init__(self, input_len, output_len, input_channels, output_channels,
                individual=False # Whether to use individual Linear layers for each channel
                ):
        assert input_channels == output_channels, "input_channels should be equal to output_channels"
        super(LLinear, self).__init__(input_len, output_len, input_channels, output_channels)
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.individual = individual

        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear.append(nn.Linear(self.input_len,self.output_len))
        else:
            self.Linear = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual==True:
            output = torch.zeros([x.size(0),self.output_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        elif self.individual==False:
            x = x.permute(0,2,1) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
            x = self.Linear(x) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, output_len)
            x = x.permute(0,2,1) # (batch_size, input_channels, output_len) -> (batch_size, output_len, input_channels)
        return x


class NLinear(TimeSeriesNN):
    """
    Normalization-Linear
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py; Modified afterwards.
    Note that input_channels == output_channels
    """
    def __init__(self, input_len, output_len, input_channels, output_channels,
                individual=False # Whether to use individual Linear layers for each channel
                ):
        assert input_channels == output_channels, "input_channels should be equal to output_channels"
        super(NLinear, self).__init__(input_len, output_len, input_channels, output_channels)
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.input_len,self.output_len))
        else:
            self.Linear = nn.Linear(self.input_len, self.output_len)

    def forward(self, x): # x shape: (batch_size, input_len, input_channels)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.output_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # x shape: (batch_size, output_len, input_channels)


class DLinear(TimeSeriesNN):
    """
    Decomposition-Linear
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py; Modified afterwards.
    Note that input_channels == output_channels
    """
    def __init__(self, input_len, output_len, input_channels, output_channels,
                individual=False, # Whether to use individual Linear layers for each channel
                kernel_size=25 # Decompsition Kernel Size
                ):
        assert input_channels == output_channels, "input_channels should be equal to output_channels"
        super(DLinear, self).__init__(input_len, output_len, input_channels, output_channels)

        class moving_avg(nn.Module):
            """
            Moving average block to highlight the trend of time series
            """
            def __init__(self, kernel_size, stride):
                super(moving_avg, self).__init__()
                self.kernel_size = kernel_size
                self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

            def forward(self, x):
                # padding on the both ends of time series
                front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
                end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
                x = torch.cat([front, x, end], dim=1)
                x = self.avg(x.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
                return x

        class series_decomp(nn.Module):
            """
            Series decomposition block
            """
            def __init__(self, kernel_size):
                super(series_decomp, self).__init__()
                self.moving_avg = moving_avg(kernel_size, stride=1)

            def forward(self, x):
                moving_mean = self.moving_avg(x)
                res = x - moving_mean
                return res, moving_mean
        
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.individual = individual
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.input_len,self.output_len))
                self.Linear_Trend.append(nn.Linear(self.input_len,self.output_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.input_len,self.output_len)
            self.Linear_Trend = nn.Linear(self.input_len,self.output_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.output_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.output_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.input_channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]