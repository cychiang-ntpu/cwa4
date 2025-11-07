import torch.nn as nn

class Conv1dBTC(nn.Conv1d):
    def forward(self, x):
        x = x.transpose(-1, -2)
        x = super().forward(x)
        x = x.transpose(-1, -2)
        return x


        
