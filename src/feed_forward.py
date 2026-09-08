import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        d_model = dimension of embeddings
        d_ff = d_ff = dimension of inner layer (feed forward neural network)
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
