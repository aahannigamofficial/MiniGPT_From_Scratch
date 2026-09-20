import torch
import torch.nn as nn


batch_size = 4
sequence_length = 4
vocab_size = 50


# Fake model predictions
logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)

# Fake target token IDs
targets = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)


print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)


# Cross-entropy loss
loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    logits.view(-1, vocab_size),
    targets.view(-1)
)

print("Loss:", loss.item())