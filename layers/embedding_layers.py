import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    '''
    (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    Positional embedding is fixed, only related to the shape of input tensor.
    '''
    def __init__(self, d_model=None, max_len=None):
        super(PositionalEmbedding, self).__init__()
        self.d_model = 64 if d_model is None else d_model
        max_len = 1000 if max_len is None else max_len
        position = np.arange(max_len)[:, np.newaxis]  # shape: (max_len, 1)
        d_model_=(d_model+1)//2*2 # 向上取偶数
        pos_enc = np.zeros((max_len, d_model_)) # 初始化位置编码矩阵
        div_term = np.exp(np.arange(0, d_model_, 2) * -(np.log(10000.0) / d_model_))
        pos_enc[:, 0::2] = np.sin(position * div_term) # even dimensions
        pos_enc[:, 1::2] = np.cos(position * div_term) # odd dimensions
        pos_enc = torch.from_numpy(pos_enc).float() # convert to tensor
        self.register_buffer('pos_enc', pos_enc) # Register a buffer that won't be updated

    def forward(self, x): # x: (batch_size, seq_len, d_model)
        pos_enc = self.pos_enc.unsqueeze(0).tile(x.size(0), 1, 1) # 扩展位置编码矩阵到batch_size维度 # pos_enc shape: (batch_size, max_len, d_model)
        return pos_enc[:, :x.size(1), :self.d_model] # 取前seq_len个位置编码 # output shape: (batch_size, seq_len, d_model)


class ConvEmbedding(nn.Module):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, d_model)
    Use 1d-CNN as token embedding approach.
    Usually output_len == input_len (tokenize each time step)
    暂时用于PatchTST
    '''
    def __init__(self, input_channels, d_model,
                kernel_size=3,
                padding=None,
                padding_mode='circular',
                stride=1,
                dilation=1,
                bias=False
                ):
        super(ConvEmbedding, self).__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding # 如果没有指定padding，则设置为使得输出序列长度不变的值
        self.tokenConv = nn.Conv1d(in_channels=input_channels,
                                    out_channels=d_model,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    padding_mode=padding_mode,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=bias)        
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
        x = self.tokenConv(x) # (batch_size, input_channels, input_len) -> (batch_size, d_model, output_len)
        x = x.permute(0, 2, 1) # (batch_size, d_model, output_len) -> (batch_size, output_len, d_model)
        return x


class TemporalEmbedding(nn.Module):
    pass