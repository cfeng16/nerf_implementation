import torch
import numpy as np
import torch.nn as nn


class positionalencoder(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.freq_list = []
        self.L = L
        for i in range(L):
            self.freq_list.append(2**i)

    def forward(self, x):
        pe = []
        pe.append(x)
        for i in self.L:
            pe.append(torch.sin(self.freq_list[i]*x))
            pe.append(torch.cos(self.freq_list[i]*x))
        pe = torch.cat(pe, dim=-1)
        return pe

