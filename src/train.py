import torch
import torch.nn as nn
import torch.optim as optim

from model import MiniGPT
from tokenizer import Tokenizer


# ============================================================
# Settings
# ============================================================

max_sequence_length = 4

embedding_dim = 8
num_heads = 2
hidden_dim = 32
num_blocks = 3

learning_rate = 0.001
epochs = 500


# ============================================================
# Load corpus
# ============================================================

with open(
    "data/corpus.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# ============================================================
# Tokenizer
# ============================================================

tokenizer = Tokenizer(text)

token_ids = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

vocab_size = len(tokenizer)

print("Vocabulary size:", vocab_size)
print("Total tokens:", len(token_ids))


# ============================================================
# Create training sequences
# ============================================================

inputs = []
targets = []

for i in range(
    len(token_ids) - max_sequence_length
):

    input_chunk = token_ids[
        i:i + max_sequence_length
    ]

    target_chunk = token_ids[
        i + 1:i + max_sequence_length + 1
    ]

    inputs.append(input_chunk)
    targets.append(target_chunk)


inputs = torch.stack(inputs)
targets = torch.stack(targets)


print("Inputs shape:", inputs.shape)
print("Targets shape:", targets.shape)


# ============================================================
# Create MiniGPT
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks
)


# ============================================================
# Loss function
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# Training loop
# ============================================================

for epoch in range(epochs):

    # Forward pass
    logits = model(inputs)

    # Calculate loss
    loss = loss_function(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Print progress
    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# Save trained model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "max_sequence_length": max_sequence_length,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token
    },
    "checkpoints/minigpt.pth"
)

print("\nModel saved to checkpoints/minigpt.pth")