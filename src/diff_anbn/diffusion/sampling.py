"""Sampling procedures for MDLM."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .mdlm import MDLM


@torch.no_grad()
def sample(
    mdlm: MDLM,
    batch_size: int,
    seq_len: int,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    show_progress: bool = True,
    initial_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate samples using ancestral sampling.

    Starts with all masked tokens and progressively unmasks them.

    Args:
        mdlm: MDLM model
        batch_size: Number of samples to generate
        seq_len: Sequence length
        num_steps: Number of denoising steps
        temperature: Sampling temperature (1.0 = normal, <1 = more greedy)
        device: Device to generate on
        show_progress: Whether to show progress bar
        initial_tokens: Optional initial tokens (partially masked)

    Returns:
        Generated tokens of shape (batch_size, seq_len)
    """
    mdlm.eval()
    device = torch.device(device)

    # Initialize with all mask tokens
    if initial_tokens is not None:
        x = initial_tokens.clone().to(device)
    else:
        x = torch.full(
            (batch_size, seq_len),
            mdlm.mask_token_id,
            dtype=torch.long,
            device=device,
        )

    # Time steps from t=0 (fully masked) to t=1 (fully unmasked)
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Sampling", leave=False)

    for i in iterator:
        t_current = timesteps[i]
        t_next = timesteps[i + 1]

        # Current masking rate and next masking rate
        mask_prob_current = mdlm.noise_schedule.mask_prob(t_current)
        mask_prob_next = mdlm.noise_schedule.mask_prob(t_next)

        # Find currently masked positions
        is_masked = x == mdlm.mask_token_id

        if not is_masked.any():
            break  # All tokens unmasked

        # Get model predictions at current time
        t_batch = t_current.expand(batch_size)
        logits = mdlm.get_logits(x, t_batch)  # (batch, seq, vocab)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Sample from predictions
        probs = F.softmax(logits, dim=-1)
        # Flatten for multinomial sampling
        flat_probs = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).view(batch_size, seq_len)

        # Determine how many tokens to unmask in this step
        # We want to go from mask_prob_current to mask_prob_next
        # The fraction of currently masked tokens to unmask:
        num_masked = is_masked.sum(dim=1).float()
        target_masked = (mask_prob_next * seq_len).expand(batch_size)

        # Number to unmask this step
        num_to_unmask = (num_masked - target_masked).clamp(min=0).long()

        # For each sample, unmask the top-confidence predictions
        for b in range(batch_size):
            if num_to_unmask[b] == 0:
                continue

            masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                continue

            # Get confidence scores for masked positions
            confidences = probs[b, masked_positions].max(dim=-1).values

            # Select top-k most confident positions to unmask
            k = min(num_to_unmask[b].item(), len(masked_positions))
            _, top_indices = confidences.topk(k)
            positions_to_unmask = masked_positions[top_indices]

            # Unmask these positions
            x[b, positions_to_unmask] = sampled[b, positions_to_unmask]

    # Final pass: unmask any remaining masked tokens
    is_masked = x == mdlm.mask_token_id
    if is_masked.any():
        t_final = torch.ones(batch_size, device=device)
        logits = mdlm.get_logits(x, t_final)
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).view(batch_size, seq_len)
        x[is_masked] = sampled[is_masked]

    return x


@torch.no_grad()
def sample_greedy(
    mdlm: MDLM,
    batch_size: int,
    seq_len: int,
    num_steps: int = 100,
    device: torch.device | str = "cpu",
    show_progress: bool = True,
) -> torch.Tensor:
    """Generate samples using greedy decoding.

    Similar to sample() but always picks the most likely token.

    Args:
        mdlm: MDLM model
        batch_size: Number of samples to generate
        seq_len: Sequence length
        num_steps: Number of denoising steps
        device: Device to generate on
        show_progress: Whether to show progress bar

    Returns:
        Generated tokens of shape (batch_size, seq_len)
    """
    return sample(
        mdlm=mdlm,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        temperature=0.0,  # Greedy
        device=device,
        show_progress=show_progress,
    )


@torch.no_grad()
def sample_with_guidance(
    mdlm: MDLM,
    batch_size: int,
    seq_len: int,
    prefix: torch.Tensor | None = None,
    suffix: torch.Tensor | None = None,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    show_progress: bool = True,
) -> torch.Tensor:
    """Generate samples with optional prefix/suffix guidance.

    Keeps specified prefix and/or suffix tokens fixed during generation.

    Args:
        mdlm: MDLM model
        batch_size: Number of samples to generate
        seq_len: Total sequence length
        prefix: Optional prefix tokens of shape (batch_size, prefix_len)
        suffix: Optional suffix tokens of shape (batch_size, suffix_len)
        num_steps: Number of denoising steps
        temperature: Sampling temperature
        device: Device to generate on
        show_progress: Whether to show progress bar

    Returns:
        Generated tokens of shape (batch_size, seq_len)
    """
    device = torch.device(device)

    # Initialize with mask tokens
    x = torch.full(
        (batch_size, seq_len),
        mdlm.mask_token_id,
        dtype=torch.long,
        device=device,
    )

    # Set prefix if provided
    fixed_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    if prefix is not None:
        prefix_len = prefix.size(1)
        x[:, :prefix_len] = prefix.to(device)
        fixed_mask[:, :prefix_len] = True

    # Set suffix if provided
    if suffix is not None:
        suffix_len = suffix.size(1)
        x[:, -suffix_len:] = suffix.to(device)
        fixed_mask[:, -suffix_len:] = True

    # Generate with fixed positions
    result = sample(
        mdlm=mdlm,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        temperature=temperature,
        device=device,
        show_progress=show_progress,
        initial_tokens=x,
    )

    # Restore fixed positions (in case sampling overwrote them)
    if prefix is not None:
        result[:, :prefix_len] = prefix.to(device)
    if suffix is not None:
        result[:, -suffix_len:] = suffix.to(device)

    return result
