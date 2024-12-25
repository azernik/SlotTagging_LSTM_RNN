# models.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

class SlotTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, dropout_prob, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings).clone().detach())

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, token_ids, lengths, tags=None):
        embeddings = self.embedding(token_ids)
        packed_embeds = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_embeds)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        rnn_out = self.dropout(rnn_out)
        emissions = self.fc(rnn_out)
        mask = (token_ids != self.embedding.padding_idx).bool()

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)
