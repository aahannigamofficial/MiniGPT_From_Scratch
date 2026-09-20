import torch
import math
import torch.nn.functional as F

X = torch.tensor([
    [0.2, 0.5, 0.1, 0.7],
    [0.8, 0.3, 0.4, 0.2],
    [0.1, 0.9, 0.6, 0.3],
    [0.5, 0.2, 0.8, 0.1]
])

embedding_dim = 4

W_Q = torch.randn(embedding_dim, embedding_dim)
W_K = torch.randn(embedding_dim, embedding_dim)
W_V = torch.randn(embedding_dim, embedding_dim)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)
scores = (Q @ K.T) / math.sqrt(embedding_dim)

print("\nAttention scores:")
print(scores)
print(scores.shape)

mask = torch.tril(torch.ones(scores.shape))

print("\nCausal mask:")
print(mask)

scores = scores.masked_fill(mask == 0, float("-inf"))

print("\nMasked scores:")
print(scores)

attention_weights = F.softmax(scores, dim=-1)

print("\nAttention weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))

output = attention_weights @ V

print("\nAttention output:")
print(output)
print(output.shape)