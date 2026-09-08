import torch
import torch.nn as nn
import torch.nn.functional as F

"""Single Head Attention"""
class Attention(nn.Module):
    def __init__(self, d_model):
        """
        d_model = dimension of embeddings
        """
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.d_model = d_model

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Last 2 dimension (-2, -1), d_model = d_Q = d_K = d_V (dimension of V)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_model ** 0.5

        if mask is not None:
            scores = scores.masked_fill(mask=mask, value=float("-inf"))

        # Apply softmax across the last dimension (dim=-1, d_model values per row), so each row sums to 1
        attention = F.softmax(scores, dim=-1)

        output = torch.matmul(attention, V)

        return output

"""
Multi-Head Attention lets the model look at the same sentence in multiple ways at the same time, 
so different heads can learn different relationships between the tokens.
"""
class MultiHeadAttenion(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model = dimension of embeddings
        num_heads = number of attention heads
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Project input into Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Combine all heads back into d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0) # q.size(0) == k.size(0) == v.size(0)
        q_seq_len = q.size(1)  # q can have a different sequence length from K/V in cross-attention
        k_seq_len = k.size(1)  # k and v have the same sequence length
        
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Split heads
        # [batch, seq_len, d_model] -> [batch, seq_len, heads, head_dim]
        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, k_seq_len, self.num_heads, self.head_dim)

        # [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5

        if mask is not None:
            scores = scores.masked_fill(mask=mask, value=float('-inf'))

        attention = F.softmax(scores, dim=-1)

        output = torch.matmul(attention, V)

        # [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
        output = output.transpose(1, 2)

        # Combine heads
        output = output.contiguous().view(batch_size, q_seq_len, self.d_model)

        output = self.W_o(output)

        return output
