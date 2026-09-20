import torch
import torch.nn as nn

embedding_dim = 8

x = torch.randn(4, embedding_dim) * 10 + 5

layer_norm = nn.LayerNorm(embedding_dim)

output = layer_norm(x)

print("Input:")
print(x)

print("\nAfter LayerNorm:")
print(output)

print("\nShape:")
print(output.shape)