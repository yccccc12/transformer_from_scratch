from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self, vocab_size=2000):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        )

    def train(self, files):
        self.tokenizer.train(files, self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id("<PAD>")

    @property
    def bos_id(self):
        return self.tokenizer.token_to_id("<BOS>")

    @property
    def eos_id(self):
        return self.tokenizer.token_to_id("<EOS>")

    @property
    def unk_id(self):
        return self.tokenizer.token_to_id("<UNK>")