import torch
import torch.nn as nn
import torch.nn.functional as F

from .mingru import MinGRU
from .layernorm import LayerNorm1d
from .conv_btc import Conv1dBTC


def create_reverse_indices(lengths, max_length=None):
    if max_length is None:
        max_length = lengths.max()
    t = torch.arange(0, max_length, device=lengths.device).view(1, -1, 1)
    r = lengths.view(-1, 1, 1) - 1 - t
    return torch.where(r >= 0, r, t)


def create_mask(lengths, max_length=None):
    if max_length is None:
        max_length = lengths.max()
    t = torch.arange(0, max_length, device=lengths.device).view(1, -1, 1)
    r = t < lengths.view(-1, 1, 1)
    return r


def random_cut(y, y_lengths, seg_length):
    B, T, K = y.shape

    if y_lengths is None:
        y_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
    else:
        y_lengths = y_lengths.to(dtype=torch.long, device=torch.device('cpu'))
    
    y_cut = torch.empty(B, seg_length, K, dtype=y.dtype, device=y.device)
    y_cut_lengths = torch.empty(B, dtype=torch.long, device=y.device)
    start_indices = (torch.rand(B) * (y_lengths - seg_length)).clamp(0).int()
    for i in range(B):
        start_index = start_indices[i]                                # inclusive
        end_index = (start_index + seg_length).clamp(0, y_lengths[i]) # not inclusive
        y_cut_length = (end_index - start_index)
        y_cut[i,:y_cut_length, :] = y[i, start_index: end_index, :]
        y_cut_lengths[i] = y_cut_length
        
    return y_cut, y_cut_lengths



class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return F.silu(self.beta * x)

class ConvSwishNorm(nn.Module):
    def __init__(self, x_dim, y_dim, kernel_size):
        super().__init__()
        self.conv = Conv1dBTC(x_dim, y_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.swish = Swish()
        self.norm = LayerNorm1d(y_dim, channels_last=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.swish(y)
        y = self.norm(y)
        return y
    
class GRUSwishNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.gru_fwd = MinGRU(x_dim, y_dim)
        self.swish = Swish()
        self.norm = LayerNorm1d(y_dim, channels_last=True)
    
    def forward(self, x):
        y, _ = self.gru_fwd(x)
        y = self.swish(y)
        y = self.norm(y)
        return y

class Classifier1(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            GRUSwishNorm(x_dim, h_dim),
            GRUSwishNorm(h_dim, h_dim),
            GRUSwishNorm(h_dim, h_dim)
        )

        self.output = nn.Linear(h_dim, 1)


    def forward(self, x):
        """_summary_

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, Ex)
            輸入過去的 GNSS + 地震能量資訊。
        Returns
        -------
        y : torch.Tensor of shape(B)  發生地震的對數機率。
        """
        h = self.encoder(x) 
        h = h[:, -1, :] # (B, C)
        h = self.output(h) # (B, 1)
        y = torch.sigmoid(h) # (B, 1)
        return y
  

