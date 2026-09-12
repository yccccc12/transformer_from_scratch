# Transformer from Scratch

## Overview
This project serves as learning purpose and implementing a Transformer from scratch using PyTorch. An encoder-decoder Transformer is implemented for English-to-French translation.

## Architecture
![Transformer Architecture](assets/architecture.png)
*Figure 1: High-level Transformer architecture*

Figure 1 shows the high-level Transformer architecture implemented in this project. It demonstrates the end-to-end process of translating a sentence, starting with tokenization, converting tokens into embeddings, and processing them through the encoder and decoder. The decoder then projects its hidden representations into vocabulary space and applies a softmax function to generate a probability distribution over all tokens in the vocabulary. The token with the highest probabilities is selected and decoded by tokenizer to produce the translated sentence.

## Dataset
The model is trained on the OPUS Books dataset — a collection of translated texts from the web, aligned by Andras Farkas. For this project, the English-French language pair is used
to train the Transformer for English-to-French translation. The dataset is available on [Hugging Face](https://huggingface.co/datasets/Helsinki-NLP/opus_books).

## Tokenization
Byte-Pair Encoding (BPE) tokenization is used to tokenize the inputs. It works by repeatedly finding the most common pairs of characters in the text and combining them into a new subword until the vocabulary reaches a desired size. However, we not going to implement BPE tokenizer from scratch in this project, instead it uses the Hugging Face `tokenizers` framework to train a BPE tokenizer on the bilingual dataset. 

The tokenizer uses a vocabulary of size 8,000 and includes four special tokens: `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`. The trained tokenizer is saved and is used to convert the source and target sentences into token IDs for transformer training and inference.

## Transformer Components
The transformer is implemented using PyTorch without using a pretrained Transformer model.

The main components are:
- Token Embeddings: Converts token IDs into dense vector representations.
- Positional Encodings: Add positional informations to the token embeddings.
- Multi-Head Attention: Allows the model to capture diverse relationships and patterns in the input sequences.
- Encoder: Processes the input sequence and produces contextual representations.
- Decoder: Generates the target sequence using the encoder output and previously generated tokens.
- Output Projection: Maps decoder representations to vocabulary-sized logits.

## Project Structure
```
transformer-from-scratch/
├── assets/                 # Architecture diagram
├── checkpoints/            # Trained tokenizer and model weights
│   ├── tokenizer.json
│   └── transformer.pt
├── data/                   # Generated training data
│   ├── train.txt           # Parallel English-French sentence pairs
│   └── corpus.txt          # Flattened text used to train the tokenizer
├── src/
│   ├── prepare_data.py     # Download/shuffle dataset, write train.txt
│   ├── train_tokenizer.py  # Build corpus.txt and train the BPE tokenizer
│   ├── tokenizer.py        # BPE tokenizer wrapper
│   ├── dataset.py          # TranslationDataset and collate function
│   ├── embedding.py        # Token embedding layer
│   ├── positional_encoding.py
│   ├── attention.py        # Single-head and multi-head attention
│   ├── feed_forward.py     # Position-wise feed-forward network
│   ├── layer_norm.py       # Layer normalization
│   ├── masks.py            # Padding and causal mask creation
│   ├── transformer.py      # Encoder, decoder, and full Transformer model
│   ├── train.py            # Training loop for Transformer model
│   ├── translate.py        # Load checkpoint and run inference
│   ├── utils.py            # Shared path constants and seeding
│   └── transformer_from_scratch.ipynb  # Standalone notebook for running the pipeline on Colab
├── requirements.txt
└── README.md
```

## Usage

### Running Locally
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
Downloads OPUS Books (en-fr), shuffles, and writes `data/train.txt`.
```bash
cd src
python prepare_data.py
```

### 3. Train the tokenizer
Builds `data/corpus.txt` from `train.txt` and trains a BPE tokenizer, saved to `checkpoints/tokenizer.json`.
```bash
python train_tokenizer.py
```

### 4. Train the model
Trains the Transformer and saves weights to `checkpoints/transformer.pt`.
```bash
python train.py
```

### 5. Translate
Loads the trained tokenizer and model, then translates a set of example English sentences into French.
```bash
python translate.py
```

> Steps 2-5 must be run in order — each step depends on files produced by the previous one. Run all commands from the `src/` directory so the relative imports (e.g. `from utils import ...`) resolve correctly.

### Running on Google Colab
If you don't have a local GPU, you can run the full pipeline in [`src/transformer_from_scratch.ipynb`](src/transformer_from_scratch.ipynb) instead — it mirrors the same steps (prepare data -> train tokenizer -> train model -> translate) in one notebook.

1. Open the notebook directly in Colab: [transformer_from_scratch.ipynb](https://colab.research.google.com/github/yccccc12/transformer_from_scratch/blob/master/src/transformer_from_scratch.ipynb).
2. Go to **Runtime -> Change runtime type** and select **T4 GPU** as the hardware accelerator.
3. Run the cells from top to bottom.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Byte-Pair Encoding (BPE) in NLP](https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/) (Geeksforgeeks, 2025)
- [Cracking the Annotated Transformer - Part I](https://drhuangxiao.com/Blogs/Cracking+the+Annotated+Transformer+-+Part+I) (Huang Xiao)
- [Parallel Data, Tools and Interfaces in OPUS](https://aclanthology.org/L12-1246/) (Tiedemann, LREC 2012)