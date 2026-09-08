import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import BPETokenizer
from dataset import TranslationDataset, create_collate_fn
from masks import create_masks
from transformer import Transformer

import random
import numpy as np

# Random Seed to ensure reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Parameters
batch_size = 32
d_model = 128
num_heads = 4
d_ff = 512
num_layers = 3
max_len = 256
learning_rate = 3e-4
num_epochs = 20

# Tokenizer
tokenizer = BPETokenizer()
tokenizer.load("data/tokenizer.json")

vocab_size = tokenizer.vocab_size
print("Vocabulary size:", vocab_size)

# Dataset
dataset = TranslationDataset("data/train.txt", tokenizer, max_len)
print("Dataset size:", len(dataset))

# DataLoader
collate_fn = create_collate_fn(tokenizer.pad_id)
loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)


# Model
model = Transformer(
    vocab_size=vocab_size,
    max_len=max_len,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

model = model.to(device)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_parameters:,}")

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.98), # Beta_1 and Beta_2
    eps=1e-9
)

# Training
for epoch in range(num_epochs):

    model.train()

    total_loss = 0.0

    for src, decoder_input, target in loader:
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        target = target.to(device)

        # Create masks
        src_mask, tgt_mask, cross_mask = create_masks(src, decoder_input, tokenizer.pad_id)

        # Forward pass
        logits = model(src, decoder_input, src_mask, tgt_mask, cross_mask)

        # logits: [B, T, vocab_size], CrossEntropyLoss expects: [N, vocab_size]
        # so flatten B and T.
        B, T, C = logits.shape

        logits = logits.reshape(B * T, C)

        target = target.reshape(B * T)

        # Calculate loss
        loss = criterion(logits, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Save model
torch.save(model.state_dict(), "data/transformer.pt")

print("Model saved to data/transformer.pt")