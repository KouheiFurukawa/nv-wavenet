import torch
import torch.nn as nn


class Autoregressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim)
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out
