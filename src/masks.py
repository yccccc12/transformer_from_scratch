import torch

def create_masks(src, tgt, pad_id):
    batch_size = src.size(0) # B
    src_len = src.size(1)    # S
    tgt_len = tgt.size(1)    # T

    # Source padding mask: We don't want encoder to pay attention to PAD
    # [B, S], False = real token, True = PAD token
    src_padding_mask = (src == pad_id)

    # [B, S] -> [B, 1, S] -> [B, 1, 1, S]
    src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)

    # Target padding mask
    # [B, T]
    tgt_padding_mask = (tgt == pad_id)

    # [B, T] -> [B, 1, 1, T]
    tgt_padding_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(2)


    # Causal mask: Used by decoder self-attention, mask future token
    # True = Block, False = Allow
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1
    )

    # [T, T] -> [1, 1, T, T]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Combine target mask
    # Decoder don't look at PAD tokens and future tokens
    # [B, 1, T, T], True if either conditions = Block
    tgt_mask = tgt_padding_mask | causal_mask

    # Cross attention mask
    # [B, 1, 1, S] -> [B, 1, T, S]

    cross_mask = src_mask.expand(batch_size, 1, tgt_len, src_len)
    
    return src_mask, tgt_mask, cross_mask