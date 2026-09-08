import torch
import torch.nn as nn
nn.LayerNorm
"""
Layer Normalization stabilizes and accelerates the training process in deep learning.

Source: 
https://www.geeksforgeeks.org/deep-learning/what-is-layer-normalization/

"""
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-5

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, correction=0)

        output = ((x - mean) / (var + self.eps) ** 0.5) * self.gamma + self.beta

        return output
