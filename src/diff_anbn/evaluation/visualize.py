"""Visualization utilities for training and evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_file: str | Path) -> list[dict]:
    """Load metrics from JSONL log file.

    Args:
        log_file: Path to metrics.jsonl file

    Returns:
        List of metric dictionaries
    """
    metrics = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def plot_accuracy_curve(
    log_file: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot syntactic accuracy over training steps.

    Args:
        log_file: Path to metrics.jsonl file
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    metrics = load_metrics(log_file)

    # Extract eval metrics
    eval_metrics = [m for m in metrics if m.get("prefix") == "eval"]

    if not eval_metrics:
        print("No evaluation metrics found in log file")
        return plt.figure()

    steps = [m["step"] for m in eval_metrics]
    accuracies = [m.get("eval/syntactic_accuracy", 0) for m in eval_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, accuracies, "b-", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="100% accuracy")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Syntactic Accuracy", fontsize=12)
    ax.set_title("Syntactic Accuracy During Training", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_loss_curve(
    log_file: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training loss over steps.

    Args:
        log_file: Path to metrics.jsonl file
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    metrics = load_metrics(log_file)

    # Extract train metrics
    train_metrics = [m for m in metrics if m.get("prefix") == "train"]

    if not train_metrics:
        print("No training metrics found in log file")
        return plt.figure()

    steps = [m["step"] for m in train_metrics]
    losses = [m.get("train/loss", 0) for m in train_metrics]

    # Smooth the loss curve
    window = min(50, len(losses) // 10 + 1)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window - 1 :]
    else:
        smoothed = losses
        smoothed_steps = steps

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, losses, "b-", alpha=0.3, linewidth=0.5, label="Raw loss")
    ax.plot(smoothed_steps, smoothed, "b-", linewidth=2, label="Smoothed loss")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_samples(
    samples: list[str],
    is_valid: list[bool],
    output_path: str | Path | None = None,
    max_samples: int = 20,
    show: bool = True,
) -> plt.Figure:
    """Visualize samples with validity annotations.

    Args:
        samples: List of decoded strings
        is_valid: List of validity flags
        output_path: Optional path to save figure
        max_samples: Maximum samples to show
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    n = min(len(samples), max_samples)
    samples = samples[:n]
    is_valid = is_valid[:n]

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.4)))

    for i, (s, valid) in enumerate(zip(samples, is_valid)):
        color = "green" if valid else "red"
        marker = "✓" if valid else "✗"
        ax.text(
            0.02,
            1 - (i + 0.5) / n,
            f"{marker} {repr(s)}",
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            color=color,
            verticalalignment="center",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    valid_count = sum(is_valid)
    ax.set_title(
        f"Generated Samples ({valid_count}/{n} valid = {valid_count/n:.1%})",
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_combined_metrics(
    log_file: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot combined training metrics (loss, accuracy, learning rate).

    Args:
        log_file: Path to metrics.jsonl file
        output_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    metrics = load_metrics(log_file)

    train_metrics = [m for m in metrics if m.get("prefix") == "train"]
    eval_metrics = [m for m in metrics if m.get("prefix") == "eval"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    ax = axes[0, 0]
    if train_metrics:
        steps = [m["step"] for m in train_metrics]
        losses = [m.get("train/loss", 0) for m in train_metrics]
        ax.plot(steps, losses, "b-", alpha=0.5, linewidth=0.5)
        # Smoothed
        window = min(50, len(losses) // 10 + 1)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], smoothed, "b-", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Training accuracy (MLM prediction accuracy)
    ax = axes[0, 1]
    if train_metrics:
        steps = [m["step"] for m in train_metrics]
        accs = [m.get("train/accuracy", 0) for m in train_metrics]
        ax.plot(steps, accs, "g-", alpha=0.5, linewidth=0.5)
        # Smoothed
        if window > 1:
            smoothed = np.convolve(accs, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], smoothed, "g-", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("MLM Prediction Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Syntactic accuracy
    ax = axes[1, 0]
    if eval_metrics:
        steps = [m["step"] for m in eval_metrics]
        accs = [m.get("eval/syntactic_accuracy", 0) for m in eval_metrics]
        ax.plot(steps, accs, "b-", marker="o", linewidth=2, markersize=4)
        ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Syntactic Accuracy (Generation)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    if train_metrics:
        steps = [m["step"] for m in train_metrics]
        lrs = [m.get("train/lr", 0) for m in train_metrics]
        ax.plot(steps, lrs, "r-", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
