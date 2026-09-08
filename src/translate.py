import torch

from tokenizer import BPETokenizer
from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
d_model = 128
num_heads = 4
d_ff = 512
num_layers = 3
max_len = 256

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load("data/tokenizer.json")

vocab_size = tokenizer.vocab_size

# Load model
model = Transformer(
    vocab_size=vocab_size,
    max_len=max_len,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
).to(device)

model.load_state_dict(
    torch.load("data/transformer.pt", map_location=device)
)

model.eval()


# Create source mask
def create_src_mask(src):
    return (src == tokenizer.pad_id).unsqueeze(1).unsqueeze(2)

# Create target causal mask
def create_tgt_mask(tgt):

    tgt_len = tgt.size(1)

    padding_mask = (tgt == tokenizer.pad_id).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    return padding_mask | causal_mask

# Translation
def translate(text):

    # Tokenize English sentence
    src_ids = tokenizer.encode(text)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    src_mask = create_src_mask(src)

    tgt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        src_embedding = model.embedding(src)
        src_embedding = model.position_encoding(src_embedding)

        encoder_output = model.encoder(src_embedding, src_mask)

        for _ in range(max_len - 1):

            tgt_mask = create_tgt_mask(tgt)

            tgt_embedding = model.embedding(tgt)
            tgt_embedding = model.position_encoding(tgt_embedding)

            decoder_output = model.decoder(tgt_embedding, encoder_output, self_mask=tgt_mask, cross_mask=src_mask)

            # Get prediction for last token
            logits = model.lm_head(decoder_output)

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # Add predicted token
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            # Stop when EOS is generated
            if next_token.item() == tokenizer.eos_id:
                break

    # Convert IDs back to text
    output_ids = tgt[0].tolist()

    # Remove BOS and EOS
    output_ids = [
        token
        for token in output_ids
        if token not in (tokenizer.bos_id, tokenizer.eos_id)
    ]

    return tokenizer.decode(output_ids)


# Test
if __name__ == "__main__":

    text = [
        "I love you.",
        "How are you?",
       " I am happy.",
        "I don't know.",
        "Where are you?",
        "I want to learn French.",
        "She is reading a book.",
        "We are going to Paris.",
        "They live in a small house.",
        "Do you speak English?"
    ]

    for txt in text:
        translation = translate(txt)
        print(f"{txt} : {translation}")
