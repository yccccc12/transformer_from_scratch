from pathlib import Path
from tokenizer import BPETokenizer

train_file = Path("data/train.txt")
corpus_file = Path("data/corpus.txt")
tokenizer_file = Path("data/tokenizer.json")

# Create corpus
with open(train_file, "r", encoding="utf-8") as f:
    with open(corpus_file, "w", encoding="utf-8") as out:

        for line in f:

            src, tgt = line.rstrip("\n").split("\t")

            out.write(src + "\n")
            out.write(tgt + "\n")

# Train BPE
tokenizer = BPETokenizer(vocab_size=8000)
tokenizer.train([str(corpus_file)])

# Save
tokenizer.save(str(tokenizer_file))

print("Tokenizer saved to:", tokenizer_file)
print("Vocabulary size:", tokenizer.vocab_size)

print("PAD:", tokenizer.pad_id) # 0
print("BOS:", tokenizer.bos_id) # 1
print("EOS:", tokenizer.eos_id) # 2
print("UNK:", tokenizer.unk_id) # 3
