from datasets import load_dataset
from utils import TRAIN_FILE

dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")

# Shuffle before selecting: the first rows of opus_books are book titles/headers, 
# not real sentences, which produced degenerate training data.
dataset = dataset.shuffle(seed=42)

dataset = dataset.select(range(20000))

count = 0

with open(str(TRAIN_FILE), "w", encoding="utf-8") as f:
    for example in dataset:
        translation = example["translation"]

        src = translation["en"].strip()
        tgt = translation["fr"].strip()

        f.write(f"{src}\t{tgt}\n")
        count += 1


print(f"Saved {count} examples to {TRAIN_FILE}")