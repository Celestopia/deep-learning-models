import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .baseclass import TimeSeriesNN

class Identical(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    使用输入序列最后一个时间步的数据作为输出序列的预测值
    用于单步预测，如果是多步预测(output_len > 1)，则输出序列在每个时间步的值一样
    可作为baseline
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels):
        assert input_channels == output_channels, "input_channels should be equal to output_channels"
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.fc=nn.Linear(1,1) # placeholder

    def forward(self, x):
        x_last_step = x[:, -1, :].view(x.shape[0], 1, x.shape[2]) # (batch_size, 1, input_channels)
        x = x_last_step.repeat(1, self.output_len, 1) # (batch_size, output_len, input_channels)
        return x


class ExponentialMovingAverage(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    使用输入序列的滑动平均作为输出序列的预测值
    用于单步预测，如果是多步预测(output_len > 1)，则输出序列在每个时间步的值一样
    通道间相互独立，对每个通道单独预测
    可作为baseline
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, alpha=None):
        assert input_channels == output_channels, "input_channels should be equal to output_channels"
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.alpha = 2/(1+input_len) if alpha is None else alpha # 如果给定了alpha，则使用给定的，否则使用默认的alpha值
        self.fc=nn.Linear(1,1) # placeholder

    def forward(self, x): # x: (batch_size, input_len, input_channels)
        ema = torch.zeros_like(x)  # 创建与输入相同形状的数组
        ema[:, 0, :] = x[:, 0, :]  # 使用输入序列的第一个值作为 EMA 的初始值
        for t in range(1, self.input_len):
            ema[:, t, :] = self.alpha * x[:, t, :] + (1 - self.alpha) * ema[:, t - 1, :]
        last_step_ema = ema[:, -1, :].view(x.shape[0], 1, x.shape[2])  # 取出最后一个时间步的 EMA 值作为输出序列的预测值
        output = last_step_ema.repeat(1, self.output_len, 1) # (batch_size, output_len, input_channels)
        return output

class ARIMA(TimeSeriesNN):
    pass


class SVR(TimeSeriesNN):#不能用，待完善
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    使用支持向量机回归作为输出序列的预测值
    对于每个多元时间序列，将其展平，以适应SVR的输入要求
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                hidden_dim=64,
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                ):
        try:
            from sklearn.svm import SVR as sklearn_SVR
        except ImportError:
            raise ImportError("sklearn is required for SVR module. Please install it using 'pip install scikit-learn' or 'conda install scikit-learn'")
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.fc1=nn.Linear(input_len*input_channels,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_len*output_channels)
        self.svr=sklearn_SVR(kernel=kernel, C=C, epsilon=epsilon)


    def forward(self, x, y=None): # x: (batch_size, input_len, input_channels)
        x = x.view(x.size(0), -1) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x = self.fc1(x) # (batch_size, input_len*input_channels) -> (batch_size, hidden_dim)
        if y is not None:
            self.svr.fit(np.array(x), np.array(y)) # fit SVR model
        x = torch.Tensor(self.svr.predict(np.array(x))) # (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        x = self.fc2(x) # (batch_size, hidden_dim) -> (batch_size, output_len*output_channels)
        x = x.view(x.shape[0], self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x