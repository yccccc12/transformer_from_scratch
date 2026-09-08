import torch.nn as nn
from embedding import Token_Embedding
from positional_encoding import Position_Encoding
from attention import MultiHeadAttenion
from layer_norm import LayerNorm
from feed_forward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        """
        d_model = dimension of embeddings
        num_heads = number of attention heads
        d_ff = dimension of inner layer (feed forward neural network)
        """
        super().__init__()

        self.attention = MultiHeadAttenion(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, x, x, mask)

        x = x + residual
        x = self.norm1(x)

        residual = x
        x = self.feed_forward(x)

        x = x + residual
        x = self.norm2(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttenion(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttenion(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        residual = x
        x = self.self_attention(x, x, x, mask=self_mask)
        x = x + residual
        x = self.norm1(x)

        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output, mask=cross_mask)
        x = x + residual
        x = self.norm2(x)

        residual = x
        x = self.feed_forward(x)
        x = x + residual
        x = self.norm3(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        for block in self.blocks:
            x = block(x, encoder_output, self_mask, cross_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, num_layers):
        """
        vocab_size = number of different tokens
        max_len = maximum sequence length
        d_model = dimension of embeddings
        num_heads = number of attention heads
        d_ff = dimension of inner layer (feed forward neural network)
        num_layer = number of stacked transformer blocks
        """
        super().__init__()

        self.embedding = Token_Embedding(vocab_size, d_model)
        self.position_encoding = Position_Encoding(max_len, d_model)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        # Encoder
        src = self.embedding(src)
        src = self.position_encoding(src)

        encoder_output = self.encoder(src, src_mask)

        # Decoder
        tgt = self.embedding(tgt)
        tgt = self.position_encoding(tgt)

        decoder_output = self.decoder(tgt, encoder_output, self_mask=tgt_mask, cross_mask=cross_mask)
        
        logits = self.lm_head(decoder_output)

        return logits
