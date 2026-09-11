from pathlib import Path
from tokenizer import BPETokenizer
from utils import TRAIN_FILE, CORPUS_FILE, TOKENIZER_FILE

# Create corpus
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    with open(CORPUS_FILE, "w", encoding="utf-8") as out:

        for line in f:

            src, tgt = line.rstrip("\n").split("\t")

            out.write(src + "\n")
            out.write(tgt + "\n")

# Train BPE
tokenizer = BPETokenizer(vocab_size=8000)
tokenizer.train([str(CORPUS_FILE)])

# Save
tokenizer.save(str(TOKENIZER_FILE))

print("Tokenizer saved to:", TOKENIZER_FILE)
print("Vocabulary size:", tokenizer.vocab_size)

print("PAD:", tokenizer.pad_id) # 0
print("BOS:", tokenizer.bos_id) # 1
print("EOS:", tokenizer.eos_id) # 2
print("UNK:", tokenizer.unk_id) # 3
