"""Masked Diffusion Language Model (MDLM) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_schedule import NoiseSchedule, CosineSchedule


class MDLM(nn.Module):
    """Masked Diffusion Language Model.

    MDLM trains a model to predict original tokens from masked inputs.
    The forward process progressively masks tokens, and the model learns
    to unmask them.

    Training:
        1. Sample t ~ U(0, 1)
        2. Mask tokens with probability (1 - alpha(t))
        3. Predict original tokens at masked positions
        4. Compute cross-entropy loss

    Sampling:
        1. Start with all tokens masked
        2. Progressively unmask based on model predictions
    """

    def __init__(
        self,
        model: nn.Module,
        noise_schedule: NoiseSchedule | None = None,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
    ):
        """Initialize MDLM.

        Args:
            model: Transformer model that takes (tokens, time) and outputs logits
            noise_schedule: Schedule for masking probability (default: cosine)
            mask_token_id: Token ID for mask token
            pad_token_id: Token ID for padding (excluded from loss)
        """
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule or CosineSchedule()
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def forward_process(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward (noising) process.

        Masks tokens with probability (1 - alpha(t)).

        Args:
            x: Original tokens of shape (batch, seq_len)
            t: Timesteps of shape (batch,) in [0, 1]

        Returns:
            Tuple of (masked_tokens, mask) where mask is True for masked positions
        """
        # Get masking probability for each sample
        mask_prob = self.noise_schedule.mask_prob(t)  # (batch,)

        # Sample mask for each position
        # mask_prob: (batch,) -> (batch, 1) for broadcasting
        rand = torch.rand_like(x.float())
        mask = rand < mask_prob.unsqueeze(-1)

        # Apply masking
        x_masked = x.clone()
        x_masked[mask] = self.mask_token_id

        return x_masked, mask

    def compute_loss(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute MDLM training loss.

        Args:
            x: Original tokens of shape (batch, seq_len)
            attention_mask: Optional mask where True/1 = valid token

        Returns:
            Dictionary with 'loss' and optional metrics
        """
        batch_size = x.size(0)
        device = x.device

        # Sample timesteps
        t = self.noise_schedule.sample_timesteps(batch_size, device)

        # Apply forward process
        x_masked, mask = self.forward_process(x, t)

        # Get model predictions
        logits = self.model(x_masked, t, attention_mask)  # (batch, seq_len, vocab)

        # Compute cross-entropy loss only on masked positions
        # Reshape for cross_entropy: (batch * seq_len, vocab) and (batch * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = x.view(-1)

        # Create loss mask: only compute loss on masked positions that aren't padding
        loss_mask = mask.view(-1)
        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask.view(-1).bool()

        # Compute loss
        if loss_mask.sum() == 0:
            # Edge case: no masked tokens
            loss = torch.tensor(0.0, device=device)
        else:
            # Select only masked positions
            logits_masked = logits_flat[loss_mask]
            targets_masked = targets_flat[loss_mask]

            loss = F.cross_entropy(logits_masked, targets_masked)

        # Compute accuracy for monitoring
        with torch.no_grad():
            if loss_mask.sum() > 0:
                preds = logits_flat[loss_mask].argmax(dim=-1)
                accuracy = (preds == targets_masked).float().mean()
            else:
                accuracy = torch.tensor(1.0, device=device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "num_masked": loss_mask.sum(),
            "mask_rate": mask.float().mean(),
        }

    def get_logits(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get model logits for given inputs.

        Args:
            x: Token IDs of shape (batch, seq_len)
            t: Timesteps of shape (batch,) in [0, 1]
            attention_mask: Optional attention mask

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.model(x, t, attention_mask)
