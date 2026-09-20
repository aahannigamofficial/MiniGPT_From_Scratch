import torch
import torch.nn as nn
embedding_dim = 8
hidden_dim = 32

ffn = nn.Sequential(
    nn.Linear(embedding_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, embedding_dim)
)
X = torch.randn(4, embedding_dim)
print("Input:", X.shape)
output = ffn(X)
print("Output:", output.shape)