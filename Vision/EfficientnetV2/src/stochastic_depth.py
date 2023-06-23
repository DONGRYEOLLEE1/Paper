import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StochasticDepth(nn.Module):
    def __init__(self, survival_prob = .8):
        super(StochasticDepth, self).__init__()
        
        self.p = survival_prob
        
    def forward(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1).to(device) < self.p
        
        return torch.div(x, self.p) * binary_tensor