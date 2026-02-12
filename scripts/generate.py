#!/usr/bin/env python3
"""Sample generation script for trained MDLM models."""

import argparse

import torch

from diff_anbn.config import ExperimentConfig
from diff_anbn.diffusion import MDLM, sample
from diff_anbn.diffusion.noise_schedule import get_schedule
from diff_anbn.languages import Tokenizer, get_language
from diff_anbn.models import DiffusionTransformer


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained MDLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Create components
    language = get_language(config.language.name)
    tokenizer = Tokenizer(vocab=language.vocab)

    model = DiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=0.0,
        max_seq_len=config.model.max_seq_len,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    noise_schedule = get_schedule(config.diffusion.noise_schedule)
    mdlm = MDLM(
        model=model,
        noise_schedule=noise_schedule,
        mask_token_id=tokenizer.mask_id,
        pad_token_id=tokenizer.pad_id,
    )

    num_steps = args.num_steps or config.diffusion.num_timesteps

    # Generate samples
    samples = sample(
        mdlm=mdlm,
        batch_size=args.num_samples,
        seq_len=config.model.max_seq_len,
        num_steps=num_steps,
        temperature=args.temperature,
        device=device,
        show_progress=True,
    )

    # Decode and print
    decoded = tokenizer.decode_batch(samples, skip_special=True)

    print("\nGenerated samples:")
    print("-" * 40)
    for i, s in enumerate(decoded):
        valid = language.validate(s)
        marker = "✓" if valid else "✗"
        print(f"{i+1:3d}. {marker} {repr(s)}")

    valid_count = sum(language.validate(s) for s in decoded)
    print("-" * 40)
    print(f"Valid: {valid_count}/{len(decoded)} ({valid_count/len(decoded):.1%})")


if __name__ == "__main__":
    main()
