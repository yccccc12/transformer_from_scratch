from pathlib import Path
import random
import numpy as np
import torch

# Constant File Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.txt"
CORPUS_FILE = DATA_DIR / "corpus.txt"

TOKENIZER_FILE = CHECKPOINT_DIR / "tokenizer.json"
MODEL_FILE = CHECKPOINT_DIR / "transformer.pt"

# Set seed to ensure reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
