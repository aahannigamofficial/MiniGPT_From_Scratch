import torch
import torch.nn as nn
vocab_size = 10
embedding_dim = 4
embedding = nn.Embedding(vocab_size, embedding_dim)
token_ids = torch.tensor([0, 1, 2, 3])
vectors = embedding(token_ids)
print(vectors)
print(vectors.shape)