import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)

        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]

        return self.dropout(x)

'''
class DataEmbedding(nn.Module):
		def init(self, c_in, d_model, dropout=0.1):
		super(DataEmbedding, self).init()
		self.value_embedding = nn.Linear(c_in, d_model)
		self.dropout = nn.Dropout(p=dropout)
		
		def forward(self, x, x_mark):
		x = x.permute(0,2,1)
		
		x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))  # [Batch, Time, d_model]
		
		
		return self.dropout(x)
   
'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DataEmbedding(nn.Module):
    def __init__(self, seg_len, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
       
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_len) d -> (b d) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d) d_model -> b d d_model', b = batch, d = ts_dim)
        
        return x_embed
        
'''
import torch
import torch.nn as nn

class DataEmbedding(nn.Module):
    def __init__(self, ts_dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.linear = nn.Linear(ts_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [Batch, Time Steps, Time Series Features]
        batch, ts_len, ts_dim = x.shape

        # Apply the linear transformation to each time step's feature
        x_embed = self.linear(x)  

        # Apply dropout and return the result
        return self.dropout(x_embed)
