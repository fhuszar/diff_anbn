"""Transformer encoder for diffusion model."""

import torch
import torch.nn as nn
from x_transformers import Encoder

from .embeddings import TokenEmbedding


class DiffusionTransformer(nn.Module):
    """Transformer encoder for masked diffusion language model.

    Uses x-transformers for the encoder backbone with custom embeddings
    that incorporate time conditioning.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        """Initialize diffusion transformer.

        Args:
            vocab_size: Size of vocabulary (including special tokens)
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension (defaults to 4 * d_model)
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if d_ff is None:
            d_ff = 4 * d_model

        # Custom embeddings with time conditioning
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # x-transformers encoder (bidirectional attention for MLM-style)
        self.encoder = Encoder(
            dim=d_model,
            depth=n_layers,
            heads=n_heads,
            ff_mult=d_ff // d_model,
            attn_dropout=dropout,
            ff_dropout=dropout,
            use_rmsnorm=True,
            ff_glu=True,
            rotary_pos_emb=False,  # We use our own positional embeddings
        )

        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len)
            t: Diffusion time of shape (batch,) with values in [0, 1]
            attention_mask: Optional attention mask of shape (batch, seq_len)
                           True/1 for positions to attend, False/0 for masked

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Get embeddings with time conditioning
        emb = self.embedding(x, t)  # (batch, seq_len, d_model)

        # Create attention mask for x-transformers (expects True for masked positions)
        mask = None
        if attention_mask is not None:
            # x-transformers expects mask where True = ignore
            mask = ~attention_mask.bool()

        # Transformer encoder
        hidden = self.encoder(emb, mask=mask)  # (batch, seq_len, d_model)

        # Project to vocabulary
        logits = self.output_proj(hidden)  # (batch, seq_len, vocab_size)

        return logits

    def get_num_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
