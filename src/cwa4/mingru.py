import torch
import torch.nn as nn
import torch.nn.functional as F

def parallel_scan(log_a, log_b):
    # log_a: (N, L, H)
    # log_b: (N, L, H)
    a_star = torch.cumsum(log_a, dim=1)
    log_h0_plus_b_star = torch.logcumsumexp(log_b - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)

def g(x):
    return torch.where(x >= 0, 
        x + 0.5, 
        x.sigmoid()
    )

def log_g(x):
    return torch.where(x >= 0, 
        torch.log(x + 0.5),
        -F.softplus(-x)
    )

class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 2 * hidden_size)
    
    def forward(self, x, prev_hidden=None):
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            input sequence, shape(N, T, C)
        prev_hidden : torch.Tensor, optional
            previous hidden state, shape(N, 1, H), by default None

        Returns
        -------
        hidden : torch.Tensor
            the output hidden layer
        next_hidden : torch.Tensor
            the last timestamp of the hidden. 
        """
        seq_len = x.shape[1]
        hidden, gate = self.linear(x).chunk(2, dim=-1)

        if seq_len == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            if prev_hidden is not None:
                #   (1 - gate) * prev_hidden + gate * hidden
                # = prev_hidden + gate * (hidden - prev_hidden) 
                out = torch.lerp(prev_hidden, hidden, gate)
            else:
                out = gate * hidden
        else:
            log_a = -F.softplus(gate) # $\log(a_t)$
            log_z = -F.softplus(-gate) # $\log(z_t)$
            log_h = log_g(hidden) # $\log(\tilde{h}_t)$
            log_b = log_z + log_h #$ \log(b_t) = \log(z_t \tilde{h}_t)$

            if prev_hidden is not None:
                log_b = torch.cat((prev_hidden.log(), log_b), dim = 1)
                log_a = F.pad(log_a, (0, 0, 1, 0))

            out = parallel_scan(log_a, log_b)
            out = out[:, -seq_len:]

        last_hidden = out[:, -1:]
        return out, last_hidden
