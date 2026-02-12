#!/usr/bin/env python3
"""Evaluation script for trained MDLM models."""

import argparse
from pathlib import Path

import torch

from diff_anbn.config import ExperimentConfig
from diff_anbn.diffusion import MDLM, sample
from diff_anbn.diffusion.noise_schedule import get_schedule
from diff_anbn.evaluation import compute_generation_metrics
from diff_anbn.evaluation.metrics import print_sample_analysis
from diff_anbn.languages import Tokenizer, get_language
from diff_anbn.models import DiffusionTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MDLM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
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

    print(f"Using device: {device}")

    # Create components
    language = get_language(config.language.name)
    tokenizer = Tokenizer(vocab=language.vocab)

    model = DiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=0.0,  # No dropout at eval
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

    print(f"\nCheckpoint from step {checkpoint['step']}")
    print(f"Best accuracy during training: {checkpoint['best_accuracy']:.4f}")

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    samples = sample(
        mdlm=mdlm,
        batch_size=args.num_samples,
        seq_len=config.model.max_seq_len,
        num_steps=config.diffusion.num_timesteps,
        temperature=args.temperature,
        device=device,
        show_progress=True,
    )

    # Decode and analyze
    decoded = tokenizer.decode_batch(samples, skip_special=True)
    print_sample_analysis(decoded, language, max_show=20)


if __name__ == "__main__":
    main()
