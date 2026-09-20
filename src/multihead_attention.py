import torch
import torch.nn as nn
import math
import torch.nn.functional as F

embedding_dim = 8
num_heads = 2

head_dim = embedding_dim // num_heads

print("Embedding dimension:", embedding_dim)
print("Number of heads:", num_heads)
print("Dimension per head:", head_dim)