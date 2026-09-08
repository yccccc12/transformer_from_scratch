import torch
import torch.nn as nn

class Position_Encoding(nn.Module):
    def __init__(self, max_len, d_model):
        """
        max_len: Max Sequence Length/No. of token
        d_model: The dimension of embedding/hidden vector
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
 
        # Create position numbers from 0 to max_len-1 and turn them into a column vector
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # Get even dimension indices
        embedding_index = torch.arange(0, d_model, 2)

        div_term = 1 / torch.tensor(10000) ** (embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Keep the tensor be part of the model, but not is fixed (not learnable) during training
        self.register_buffer('pe', pe)


    def forward(self, embeddings):
        """
        embeddings shape: (batch_size, seq_len, d_model)
        pe shape: (seq_len, d_model)
        """
        return embeddings + self.pe[:embeddings.size(1), :]

    
"""
   Token: I love cats
Position: 0   1    2

unsqueeze(1)
[[0],
 [1],
 [2]]

embedding_index = torch.arange(0, d_model, 2)

Suppose d_model = 4
embeddnig_index -> [0, 2] (2i)

Sample Usecase:

embeddings =
[
  [0.2, 0.7, 0.1, 0.9],   # I
  [0.4, 0.3, 0.8, 0.2],   # love
  [0.6, 0.5, 0.9, 0.1]    # cats
]

PE =
[
  [0.000, 1.000, 0.000, 1.000],   # position 0
  [0.841, 0.540, 0.010, 1.000],   # position 1
  [0.909,-0.416, 0.020, 1.000]    # position 2
]

embeddings + PE =
[
  [0.200, 1.700, 0.100, 1.900],
  [1.241, 0.840, 0.810, 1.200],
  [1.509, 0.084, 0.920, 1.100]
]
"""