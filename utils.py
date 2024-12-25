# utils.py
import torch
import numpy as np
import random
from typing import Dict

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_embedding_matrix(vocab: Dict[str, int], word_vectors, embedding_dim: int = 300):
    """
    Creates an embedding matrix using pretrained word vectors.
    """
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
    return torch.tensor(embedding_matrix, dtype=torch.float)