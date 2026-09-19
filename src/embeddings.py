import torch 
import torch.nn as nn 

vocab_size = 10
embedding_dim = 4
max_sequence_length = 8

token_embedding = nn.Embedding(vocab_size,embedding_dim)
position_embedding = nn.Embedding(max_sequence_length,embedding_dim)

token_ids = torch.tensor([0,1,2,3])
positions = torch.arange(len(token_ids))

token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(positions)

final_vectors = token_vectors + position_vectors

print("Token vectors:")
print(token_vectors)

print("\nPosition vectors:")
print(position_vectors)

print("\nFinal vectors:")
print(final_vectors)

print("\nShape:")
print(final_vectors.shape)