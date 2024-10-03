import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .baseclass import TimeSeriesNN
from blitz.utils import variational_estimator # 需要安装库：conda install -c conda-forge blitz-bayesian-pytorch


@variational_estimator
class BNN_conv1d(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels,
                kernel_size=3,
                padding=1
                ):
        try:
            from blitz.modules import BayesianLinear, BayesianConv1d
        except ImportError:
            raise ImportError("Please install blitz-bayesian-pytorch to use BNN_conv1d.\nCommand: conda install -c conda-forge blitz-bayesian-pytorch")
        super().__init__(input_len, output_len, input_channels, output_channels,)
        self.input_len = input_len
        self.output_len = output_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = BayesianConv1d(in_channels=input_channels, out_channels=input_channels ,kernel_size=kernel_size, padding=padding)
        self.conv2 = BayesianConv1d(in_channels=input_channels, out_channels=input_channels ,kernel_size=kernel_size, padding=padding)
        self.fc=BayesianLinear(in_features=input_channels*(input_len//4), out_features=output_channels*output_len)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
        x = self.conv1(x) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, input_len)
        x = nn.MaxPool1d(kernel_size=2)(x) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, input_len//2)
        x = nn.functional.relu(x)
        x = self.conv2(x) # (batch_size, input_channels, input_len//2) -> (batch_size, input_channels, input_len//2)
        x = nn.MaxPool1d(kernel_size=2)(x) # (batch_size, input_channels, input_len//2) -> (batch_size, input_channels, input_len//4)
        x = x.view(x.size(0), -1) # (batch_size, input_channels, input_len//4) -> (batch_size, input_channels*input_len//4)
        x = self.fc(x) # (batch_size, input_channels*input_len//4) -> (batch_size, output_channels*output_len)
        x = x.view(x.size(0), self.output_len, self.output_channels) # (batch_size, output_channels*output_len) -> (batch_size, output_len, output_channels)
        return x

    def predict(self, input: torch.Tensor,
                num_experiments= 100 # 对于同一输入的预测次数
                ):
        '''
        input: (batch_size, input_len, input_channels)
        使用蒙特卡洛方法对同一输入进行预测，返回预测结果的均值和方差
        '''
        import tqdm
        assert input.dim()==3
        self.eval()
        Y_predicted_list=[]
        for _ in tqdm.tqdm(range(num_experiments)):
            Y_predicted=self(input).detach().numpy().reshape(input.shape[0], self.output_len, self.output_channels) # Y_predicted: shape: (batch_size, output_len, output_channels); type: numpy.ndarray
            Y_predicted_list.append(Y_predicted)
        Y_predicted_list=np.array(Y_predicted_list) # Y_predicted_list: (num_experiments, batch_size, output_len, output_channels)
        Y_mean = np.mean(Y_predicted_list, axis=0) # Y_mean: shape: (batch_size, output_len, output_channels)
        Y_std = np.std(Y_predicted_list, axis=0) # Y_std: shape: (batch_size, output_len, output_channels)
        return Y_mean, Y_std