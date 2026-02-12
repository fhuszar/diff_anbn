"""Embedding modules for diffusion transformer."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512):
        """Initialize positional embeddings.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional embeddings added
        """
        return x + self.pe[:, : x.size(1)]


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps.

    Maps continuous time t in [0, 1] to embedding vectors.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0):
        """Initialize time embedding.

        Args:
            d_model: Embedding dimension
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        # MLP to transform sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings.

        Args:
            t: Time tensor of shape (batch,) with values in [0, 1]

        Returns:
            Time embeddings of shape (batch, d_model)
        """
        # Scale time to match typical timestep ranges
        t = t * 1000.0

        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )

        # Outer product: (batch, 1) * (half_dim,) -> (batch, half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)

        # Concatenate sin and cos
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If d_model is odd, pad with zero
        if self.d_model % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return self.mlp(embedding)


class TokenEmbedding(nn.Module):
    """Token embedding with optional time conditioning."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize token embedding.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.time_embedding = TimeEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        # Layer norm after combining embeddings
        self.norm = nn.LayerNorm(d_model)

        # Scale embeddings
        self.scale = math.sqrt(d_model)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute combined embeddings.

        Args:
            x: Token IDs of shape (batch, seq_len)
            t: Optional time tensor of shape (batch,) with values in [0, 1]

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        # Token embeddings
        emb = self.token_embedding(x) * self.scale

        # Add positional embeddings
        emb = self.pos_embedding(emb)

        # Add time conditioning (broadcast across sequence)
        if t is not None:
            time_emb = self.time_embedding(t)  # (batch, d_model)
            emb = emb + time_emb.unsqueeze(1)  # broadcast to (batch, seq_len, d_model)

        emb = self.norm(emb)
        emb = self.dropout(emb)

        return emb
