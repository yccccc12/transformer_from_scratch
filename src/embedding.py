import torch.nn as nn

class Token_Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        vocab_size: number of different tokens
        embedding_dim: number of values used to represent each token
        """
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, token_ids):
        """
        token_ids: token IDs representing the input text. Each ID corresponds to a token in the vocabulary.
        """
        return self.embedding(token_ids)
    
