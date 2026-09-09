# Transformer from Scratch

## Overview
This project serves as learning purpose and building transformer from scratch using pytorch. The types of transformer being implemented is encoder-decoder transformer where it perform English to French translation.

## Architecture
![Transformer Architecture](assets/architecture.png)
*Figure 1: High-level Transformer architecture*

Figure 1 shows the high-level Transformer architecture implemented in this project. It demonstrates the end-to-end process of translating a sentence, starting with tokenization, converting tokens into embeddings, and processing them through encoder and decoder. The decoder then projects its hidden representations into vocabulary space and applies a softmax function to generate probability distribution over all tokens in the vocabulary. The token with highest probabilities is selected and decoded by tokenizer to produce the translated sentence.

## Datasets
The model is trained on the OPUS Books dataset — a collection of translated texts from the web, aligned by Andras Farkas. For this project, the English-French language pair is used
to trian the Tranformer for English-to-French translation. The datasets is available at [huggingface](https://huggingface.co/datasets/Helsinki-NLP/opus_books).

## Tokenization
*To be continue*

### References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Cracking the Annotated Transformer - Part I](https://drhuangxiao.com/Blogs/Cracking+the+Annotated+Transformer+-+Part+I) (Huang Xiao)
- [Parallel Data, Tools and Interfaces in OPUS](https://aclanthology.org/L12-1246/) (Tiedemann, LREC 2012)