import torch
import torch.nn as nn
import math
import torch.nn.functional as F
embedding_dim = 8
num_heads = 2
head_dim = embedding_dim // num_heads

X = torch.randn(4, embedding_dim)

print("Original:", X.shape)

W_Q = torch.randn(embedding_dim, embedding_dim)
W_K = torch.randn(embedding_dim, embedding_dim)
W_V = torch.randn(embedding_dim, embedding_dim)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q before splitting:", Q.shape)
print("K before splitting:", K.shape)
print("V before splitting:", V.shape)


Q = Q.view(4, num_heads, head_dim).transpose(0, 1)
K = K.view(4, num_heads, head_dim).transpose(0, 1)
V = V.view(4, num_heads, head_dim).transpose(0, 1)

print("Q after splitting:", Q.shape)
print("K after splitting:", K.shape)
print("V after splitting:", V.shape)

scores = Q @ K.transpose(-2, -1)

print("Scores:", scores.shape)

scores = scores / math.sqrt(head_dim)
mask = torch.tril(torch.ones(4, 4))
scores = scores.masked_fill(mask == 0, float("-inf"))
attention_weights = F.softmax(scores, dim=-1)

print("Attention weights:", attention_weights.shape)

output = attention_weights @ V

print("Output before combining heads:", output.shape)

output = output.transpose(0, 1)
output = output.contiguous().view(4, embedding_dim)
print("After combining heads:", output.shape)

output_projection = nn.Linear(embedding_dim, embedding_dim)
output = output_projection(output)
print("Final output:", output.shape)