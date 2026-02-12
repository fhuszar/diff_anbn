#!/usr/bin/env python3
"""Training script for MDLM on formal languages."""

import argparse
from pathlib import Path

from diff_anbn.config import ExperimentConfig
from diff_anbn.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train MDLM on formal languages")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Loaded config: {config.name}")
    print(f"Language: {config.language.name}")
    print(f"Model: d={config.model.d_model}, L={config.model.n_layers}, H={config.model.n_heads}")
    print(f"Training: {config.training.num_steps} steps, batch_size={config.training.batch_size}")

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
