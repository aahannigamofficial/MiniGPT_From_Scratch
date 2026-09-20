import torch
import torch.nn as nn

from transformer import TransformerStack


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_blocks
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

        # Transformer blocks
        self.transformer = TransformerStack(
            num_blocks,
            embedding_dim,
            num_heads,
            hidden_dim
        )

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # Language modeling head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        # token_ids:
        # [batch_size, sequence_length]

        batch_size, sequence_length = token_ids.shape

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_vectors = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine token + position information
        # ----------------------------------------------------

        X = token_vectors + position_vectors

        # ----------------------------------------------------
        # Transformer stack
        # ----------------------------------------------------

        X = self.transformer(X)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        X = self.final_layer_norm(X)

        # ----------------------------------------------------
        # Language modeling head
        # ----------------------------------------------------

        logits = self.lm_head(X)

        return logits