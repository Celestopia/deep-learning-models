import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .baseclass import TimeSeriesNN


class PositionalEmbedding(nn.Module):
    '''
    (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    Positional embedding is fixed, only related to the shape of input tensor.
    '''
    def __init__(self, d_model=None, max_len=None):
        super(PositionalEmbedding, self).__init__()
        d_model = 64 if d_model is None else d_model
        max_len = 5000 if max_len is None else max_len
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
        return pos_enc[:, :x.size(1), :x.size(2)] # 取前seq_len个位置编码 # output shape: (batch_size, seq_len, d_model)


class ConvEmbedding(nn.Module):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, d_model)
    Use 1d-CNN as token embedding approach.
    Usually output_len == input_len (tokenize each time step)
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


class Transformer(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    An Encoder-only vanilla Transformer
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, 
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True # whether to use batch_size as the first dimension
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.embedding_layer = nn.Linear(input_channels, d_model) # Token embedding layer
        self.pos_emb=PositionalEmbedding(d_model, max_len=input_len) # Positional embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=drop_out, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1=nn.Linear(d_model, input_channels)
        self.fc2=nn.Linear(input_len*input_channels, output_len*output_channels)

    def forward(self, x):
        x=self.embedding_layer(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, d_model)
        x=x+self.pos_emb(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, d_model)
        x=self.transformer_encoder(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, d_model) # dimension does not change
        x=self.fc1(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, input_channels)
        x=nn.functional.relu(x)
        x=x.view(-1, self.input_len*self.input_channels) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x=self.fc2(x) # (batch_size, input_len*input_channels) -> (batch_size, output_len*output_channels)
        x=x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x

# 定义iTransformer模型
class iTransformer(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    An Encoder-only iTransformer
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, 
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True # whether to use batch_size as the first dimension
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.embedding_layer = nn.Linear(input_len, d_model) # Token embedding layer
        self.pos_emb=PositionalEmbedding(d_model, max_len=input_len) # Positional embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=drop_out, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1=nn.Linear(d_model, input_len)
        self.fc2=nn.Linear(input_len*input_channels, output_len*output_channels)

    def forward(self, x):
        x=x.permute(0, 2, 1) # (batch_size, input_len, input_channels) -> (batch_size, input_channels, input_len)
        x=self.embedding_layer(x) # (batch_size, input_channels, input_len) -> (batch_size, input_channels, d_model)
        x=x+self.pos_emb(x) # (batch_size, input_channels, d_model) -> (batch_size, input_channels, d_model)
        x=self.transformer_encoder(x) # (batch_size, input_channels, d_model) -> (batch_size, input_channels, d_model) # dimension does not change
        x=self.fc1(x) # (batch_size, input_channels, d_model) -> (batch_size, input_channels, input_len)
        x=nn.functional.relu(x)
        x=x.view(-1, self.input_len*self.input_channels) # (batch_size, input_channels, input_len) -> (batch_size, input_len*input_channels)
        x=self.fc2(x) # (batch_size, input_len*input_channels) -> (batch_size, output_len*output_channels)
        x=x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x


class PatchTST(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    An Encoder-only PatchTST
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, 
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True, # whether to use batch_size as the first dimension
                patch_size=5,
                patch_stride=1,
                ):
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.embedding_layer = ConvEmbedding(input_channels, d_model, kernel_size=patch_size, stride=patch_stride) # Token embedding layer
        self.pos_emb = PositionalEmbedding(d_model, max_len=input_len) # Positional embedding layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=drop_out, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1=nn.Linear(d_model, input_channels)
        self.fc2=nn.Linear(input_len*input_channels, output_len*output_channels)

    def forward(self, x):
        x=self.embedding_layer(x) # (batch_size, input_len, input_channels) -> (batch_size, input_len, d_model)
        x=x+self.pos_emb(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, d_model)
        x=self.transformer_encoder(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, d_model) # dimension does not change
        x=self.fc1(x) # (batch_size, input_len, d_model) -> (batch_size, input_len, input_channels)
        x=nn.functional.relu(x)
        x=x.view(-1, self.input_len*self.input_channels) # (batch_size, input_len, input_channels) -> (batch_size, input_len*input_channels)
        x=self.fc2(x) # (batch_size, input_len*input_channels) -> (batch_size, output_len*output_channels)
        x=x.view(-1, self.output_len, self.output_channels) # (batch_size, output_len*output_channels) -> (batch_size, output_len, output_channels)
        return x