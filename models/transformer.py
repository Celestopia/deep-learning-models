import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .baseclass import TimeSeriesNN
from layers.embedding_layers import PositionalEmbedding, ConvEmbedding
from layers.transformer_layers import Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer
from layers.attention_layers import AttentionLayer, ProbAttention, ReformerLayer


class Transformer(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    An Encoder-only vanilla Transformer
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True, # whether to use batch_size as the first dimension
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.

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


class iTransformer(TimeSeriesNN):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    An Encoder-only iTransformer
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels, *args, 
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True, # whether to use batch_size as the first dimension
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        
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
    def __init__(self, input_len, output_len, input_channels, output_channels, *args,
                num_layers=6, # Number of encoder layers
                d_model=64, # dimension of hidden layer
                nhead=8, # number of heads in multi-head attention
                dim_feedforward=256, # middle layer dimension in feed-forward network
                drop_out=0.1, # dropout rate
                batch_first=True, # whether to use batch_size as the first dimension
                patch_size=5, # the size in time dimension of each patch
                patch_stride=1, # the stride in time dimension of each patch
                **kwargs
                ): # For compatibility, we allow extra arguments here, but be sure they are not used.

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



# 暂时能跑，不要动 # can run, don't modify
class Reformer(TimeSeriesNN):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    Source: https://github.com/thuml/iTransformer/blob/main/model/Reformer.py
    """

    def __init__(self, input_len, output_len, input_channels, output_channels, label_len, *args,
                    num_layers=6, # Number of encoder layers
                    d_model=64, # dimension of hidden layer
                    nhead=8, # number of heads in multi-head attention
                    dim_feedforward=256, # middle layer dimension in feed-forward network
                    drop_out=0.1, # dropout rate
                    batch_first=True, # whether to use batch_size as the first dimension
                    bucket_size=4,
                    n_hashes=4,
                    **kwargs
                    ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        """
        (Make sure your screen is wide enough to display the annotation below correctly)

                         |<----------input_len------------------->|
        Input sequence:  ├────────────────────────────────────────┤
        Label sequence:                          ├────────────────┤
        Target sequence:                         ├───────────────────────────────────────────────┤
                                                 |<--label_len--->|<----------pred_len---------->|
                                                 |<-----------------output_len------------------>|
        Time axis------------------------------------------------------------------------------------------->
        """
        super().__init__(input_len, output_len, input_channels, output_channels)
        self.label_len=label_len
        self.pred_len=output_len-label_len
        self.data_emb=nn.Linear(input_channels, d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model, max_len=input_len+output_len)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(attention=None,
                                  d_model=d_model,
                                  n_heads=nhead,
                                  bucket_size=bucket_size,
                                  n_hashes=n_hashes),
                    d_model=d_model,
                    d_ff=dim_feedforward,
                    dropout=drop_out,
                    activation="relu"
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(
            d_model, output_channels, bias=True)


    def forward(self, x, y, mask=None):
        # x shape: (batch_size, input_len, input_channels)
        # y shape: (batch_size, output_len, output_channels)
        # input_channels == output_channels
        assert x.shape[2] == y.shape[2], "Input and output channels must be equal"
        x = torch.cat([x, y[:, -self.pred_len:, :]], dim=1)
        enc_out = self.data_emb(x)+self.pos_emb(x)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out # (batch_size, output_len, output_channels)


# 暂时能跑，不要动 # can run, don't modify
class Informer(TimeSeriesNN):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    Source: https://github.com/thuml/iTransformer/blob/main/model/Informer.py
    """

    def __init__(self, input_len, output_len, input_channels, output_channels, label_len, *args,
                    num_layers=6, # Number of encoder layers
                    d_model=64, # dimension of hidden layer
                    nhead=8, # number of heads in multi-head attention
                    dim_feedforward=256, # middle layer dimension in feed-forward network
                    drop_out=0.1, # dropout rate
                    batch_first=True, # whether to use batch_size as the first dimension
                    **kwargs
                    ): # For compatibility, we allow extra arguments here, but be sure they are not used.
        """
        (Make sure your screen is wide enough to display the annotation below correctly)

                         |<----------input_len------------------->|
        Input sequence:  ├────────────────────────────────────────┤
        Label sequence:                          ├────────────────┤
        Target sequence:                         ├───────────────────────────────────────────────┤
                                                 |<--label_len--->|<----------pred_len---------->|
                                                 |<-----------------output_len------------------>|
        Time axis------------------------------------------------------------------------------------------->
        """
        super(Informer, self).__init__(input_len, output_len, input_channels, output_channels)
        self.label_len = label_len
        self.pred_len=output_len-label_len

        # Embedding
        self.enc_embedding = nn.Linear(self.input_channels, d_model)
        self.dec_embedding = nn.Linear(self.output_channels, d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model, max_len=input_len+output_len)
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(mask_flag=False,
                                      factor=5,
                                      attention_dropout=drop_out,
                                      output_attention=False),
                        d_model=d_model, n_heads=nhead),
                    d_model=d_model,
                    d_ff=dim_feedforward,
                    dropout=drop_out,
                    activation="relu"
                ) for _ in range(num_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(mask_flag=True,
                                      factor=5,
                                      attention_dropout=drop_out,
                                      output_attention=False),
                        d_model=d_model, n_heads=nhead),
                    AttentionLayer(
                        ProbAttention(mask_flag=False,
                                      factor=5,
                                      attention_dropout=drop_out,
                                      output_attention=False),
                        d_model=d_model, n_heads=nhead),
                    d_model=d_model,
                    d_ff=dim_feedforward,
                    dropout=drop_out,
                    activation="relu",
                )
                for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, output_channels, bias=True)
        )

    def forward(self, x_enc, x_dec, mask=None):
        enc_out = self.enc_embedding(x_enc)+self.pos_emb(x_enc)
        dec_out = self.dec_embedding(x_dec)+self.pos_emb(x_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out # (batch_size, output_len, output_channels)