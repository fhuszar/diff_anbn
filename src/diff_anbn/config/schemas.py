"""Pydantic configuration schemas."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    """Configuration for formal language."""

    name: str = Field(description="Language name (e.g., 'anbn', 'dyck')")
    min_n: int = Field(default=1, description="Minimum generation parameter")
    max_n: int = Field(default=15, description="Maximum generation parameter")


class ModelConfig(BaseModel):
    """Configuration for transformer model."""

    d_model: int = Field(default=128, description="Model dimension")
    n_heads: int = Field(default=4, description="Number of attention heads")
    n_layers: int = Field(default=4, description="Number of transformer layers")
    d_ff: int | None = Field(default=None, description="Feed-forward dimension (4*d_model if None)")
    dropout: float = Field(default=0.1, description="Dropout probability")
    max_seq_len: int = Field(default=64, description="Maximum sequence length")


class DiffusionConfig(BaseModel):
    """Configuration for diffusion process."""

    noise_schedule: Literal["linear", "cosine", "sigmoid"] = Field(
        default="cosine", description="Noise schedule type"
    )
    num_timesteps: int = Field(default=100, description="Number of sampling timesteps")


class TrainingConfig(BaseModel):
    """Configuration for training."""

    batch_size: int = Field(default=64, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    num_steps: int = Field(default=50000, description="Total training steps")
    warmup_steps: int = Field(default=1000, description="Learning rate warmup steps")
    grad_clip: float = Field(default=1.0, description="Gradient clipping norm")
    eval_every: int = Field(default=1000, description="Evaluate every N steps")
    save_every: int = Field(default=5000, description="Save checkpoint every N steps")
    log_every: int = Field(default=100, description="Log metrics every N steps")
    num_eval_samples: int = Field(default=256, description="Number of samples for evaluation")
    seed: int = Field(default=42, description="Random seed")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    name: str = Field(description="Experiment name")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    language: LanguageConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    device: str = Field(default="auto", description="Device ('auto', 'cpu', 'cuda', 'mps')")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_device(self) -> str:
        """Get the actual device to use."""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device
