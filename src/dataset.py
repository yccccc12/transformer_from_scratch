import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

"""
Data was stored in training file and each line contain:
{English sentence} <TAB> {French sentence}
"""
class TranslationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.data = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # src_text = English, tgt_text = French
                src_text, tgt_text = line.split("\t")

                # Convert text to token
                src_ids = tokenizer.encode(src_text)
                tgt_ids = tokenizer.encode(tgt_text)

                # Skip if english sentence > maximum sequence length
                if len(src_ids) > max_len:
                    continue

                # <BOS> and <EOS> will be added to target text (French), so + 2
                if len(tgt_ids) + 2 > max_len:
                    continue

                # Add special tokens to target text
                # E.g. [Je, vous, aime] -> [<BOS>, Je, vous, aime, <EOS>]
                # <BOS> = Begining of Sentence, <EOS> = End of Sentence
                tgt_ids = [tokenizer.bos_id, *tgt_ids, tokenizer.eos_id]

                self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]

        src = torch.tensor(src_ids, dtype=torch.long)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)

        decoder_input = tgt[:-1] # Remove the final <EOS>
        target_output = tgt[1:]  # Remove the first <BOS>

        # English input, French decoder input, Expected French ouput
        return src, decoder_input, target_output


"""
Function that will prepare batches by padding sentences to the same length.

Sentences have different lengths, <PAD> tokens is added so that they will end up
with same lengths.

[10, 20, 30] -> [10, 20, 30]
[10, 20]     -> [10, 20, PAD]
"""
def create_collate_fn(pad_id):

    def collate_fn(batch):
        src, decoder_input, target_output = zip(*batch)

        # Add <PAD> tokens
        src = pad_sequence(src, batch_first=True, padding_value=pad_id)

        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=pad_id)

        target_output = pad_sequence(target_output, batch_first=True, padding_value=pad_id)

        return src, decoder_input, target_output

    return collate_fn
