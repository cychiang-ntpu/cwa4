import torch.nn as nn

class LayerNorm1d(nn.GroupNorm):
    def __init__(self, num_channels, eps=1e-05, affine=True, channels_last=False, device=None, dtype=None):
        super().__init__(num_channels, num_channels, eps, affine, device, dtype)
        self.btc = channels_last
    
    def forward(self, x):
        old_size = x.size()

        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.btc:
            x = x.transpose(1, 2)
        
        x = super().forward(x)

        if self.btc:
            x = x.transpose(1, 2)

        x = x.reshape(old_size)

        return x

