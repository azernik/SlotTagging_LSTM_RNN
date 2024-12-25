# datasets.py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple


class SlotTaggingDataset(Dataset):
    def __init__(self, data, token_vocab=None, tag_vocab=None, training=True):
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}
            for _, row in data.iterrows():
                tokens = row['utterances'].split()
                tags = row['IOB Slot tags'].split() if 'IOB Slot tags' in row else None
                for token in tokens:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                if tags:
                    for tag in tags:
                        if tag not in self.tag_vocab:
                            self.tag_vocab[tag] = len(self.tag_vocab)
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        self.corpus_token_ids = [
            torch.tensor([self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in row['utterances'].split()])
            for _, row in data.iterrows()
        ]
        
        if 'IOB Slot tags' in data.columns:
            self.corpus_tag_ids = [
                torch.tensor([
                    self.tag_vocab.get(tag, self.tag_vocab['<PAD>'])  # Map unknown tags to <PAD>
                    for tag in row['IOB Slot tags'].split()
                ])
                for _, row in data.iterrows()
            ]
        else:
            self.corpus_tag_ids = None

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        if self.corpus_tag_ids is not None:
            return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]
        else:
            return self.corpus_token_ids[idx], idx



def collate_fn(batch):
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] if isinstance(item[1], torch.Tensor) else torch.tensor([]) for item in batch]
    lengths = [len(seq) for seq in token_ids]
    max_length = max(len(seq) for seq in token_ids)

    tokens_padded = torch.stack([
        F.pad(seq, (0, max_length - len(seq)), value=0) for seq in token_ids
    ])
    tags_padded = torch.stack([
        F.pad(seq, (0, max_length - len(seq)), value=0) if len(seq) > 0 else torch.full((max_length,), 0)
        for seq in tag_ids
    ])
    return tokens_padded, tags_padded, lengths


def ids_to_tags(tag_ids: List[List[int]], tag_vocab: Dict[int, str]) -> List[List[str]]:
    """ Converts tag IDs to tag strings """
    id_to_tag = {id_: tag for tag, id_ in tag_vocab.items()}
    return [[id_to_tag[id_] for id_ in seq] for seq in tag_ids]


def collate_fn_inference(batch):
    token_ids = [item[0] for item in batch]
    lengths = [len(seq) for seq in token_ids]
    max_length = max(len(seq) for seq in token_ids)

    tokens_padded = torch.stack([
        F.pad(seq, (0, max_length - len(seq)), value=0) for seq in token_ids
    ])

    return tokens_padded, lengths