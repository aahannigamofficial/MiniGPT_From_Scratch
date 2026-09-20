import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.W_Q = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W_K = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W_V = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, X):

        batch_size, sequence_length, _ = X.shape

        # ----------------------------------------------------
        # Create Q, K and V
        # ----------------------------------------------------

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # ----------------------------------------------------
        # Split into multiple heads
        # ----------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(self.head_dim)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        output = attention_weights @ V

        # ----------------------------------------------------
        # Combine heads
        # ----------------------------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # ----------------------------------------------------
        # Final projection
        # ----------------------------------------------------

        output = self.output_projection(output)

        return output


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, X):

        return self.network(X)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):

        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

    def forward(self, X):

        # ----------------------------------------------------
        # Self-attention + residual connection
        # ----------------------------------------------------

        attention_output = self.attention(
            self.layer_norm_1(X)
        )

        X = X + attention_output

        # ----------------------------------------------------
        # Feed-forward network + residual connection
        # ----------------------------------------------------

        feed_forward_output = self.feed_forward(
            self.layer_norm_2(X)
        )

        X = X + feed_forward_output

        return X


class TransformerStack(nn.Module):

    def __init__(
        self,
        num_blocks,
        embedding_dim,
        num_heads,
        hidden_dim
    ):

        super().__init__()

        self.blocks = nn.ModuleList([

            TransformerBlock(
                embedding_dim,
                num_heads,
                hidden_dim
            )

            for _ in range(num_blocks)
        ])

    def forward(self, X):

        for block in self.blocks:

            X = block(X)

        return X