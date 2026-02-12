"""Noise schedules for masked diffusion."""

import math
from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules.

    A noise schedule defines alpha(t) where:
    - alpha(0) = 0 (fully masked at t=0)
    - alpha(1) = 1 (fully unmasked at t=1)

    The masking probability at time t is (1 - alpha(t)).
    """

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha(t) for given timesteps.

        Args:
            t: Timesteps in [0, 1], shape (batch,) or scalar

        Returns:
            Alpha values in [0, 1], same shape as t
        """
        pass

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Compute masking probability at time t.

        Args:
            t: Timesteps in [0, 1]

        Returns:
            Masking probability (1 - alpha(t))
        """
        return 1.0 - self.alpha(t)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, 1].

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        return torch.rand(batch_size, device=device)


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule: alpha(t) = t."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Linear schedule: alpha(t) = t."""
        return t


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule from improved DDPM.

    alpha(t) = cos((1-t) * pi/2)^2

    This gives slower masking near t=1 (data) and t=0 (noise),
    with faster transition in the middle.
    """

    def __init__(self, s: float = 0.008):
        """Initialize cosine schedule.

        Args:
            s: Small offset to prevent alpha(0) from being exactly 0
        """
        self.s = s

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Cosine schedule."""
        # f(t) = cos^2((t + s) / (1 + s) * pi/2)
        # Normalize so f(0) = 0 and f(1) = 1
        f_t = torch.cos((1 - t + self.s) / (1 + self.s) * math.pi / 2) ** 2
        f_0 = math.cos((1 + self.s) / (1 + self.s) * math.pi / 2) ** 2
        f_1 = math.cos(self.s / (1 + self.s) * math.pi / 2) ** 2

        # Normalize to [0, 1]
        return (f_t - f_0) / (f_1 - f_0)


class SigmoidSchedule(NoiseSchedule):
    """Sigmoid noise schedule.

    Provides smooth S-curve transition.
    """

    def __init__(self, start: float = -3.0, end: float = 3.0):
        """Initialize sigmoid schedule.

        Args:
            start: Sigmoid input at t=0
            end: Sigmoid input at t=1
        """
        self.start = start
        self.end = end

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Sigmoid schedule."""
        # Linear interpolation in sigmoid input space
        x = self.start + t * (self.end - self.start)
        sig = torch.sigmoid(x)

        # Normalize to [0, 1]
        sig_start = torch.sigmoid(torch.tensor(self.start))
        sig_end = torch.sigmoid(torch.tensor(self.end))

        return (sig - sig_start) / (sig_end - sig_start)


def get_schedule(name: str, **kwargs) -> NoiseSchedule:
    """Get noise schedule by name.

    Args:
        name: Schedule name ('linear', 'cosine', 'sigmoid')
        **kwargs: Additional arguments for the schedule

    Returns:
        NoiseSchedule instance
    """
    schedules = {
        "linear": LinearSchedule,
        "cosine": CosineSchedule,
        "sigmoid": SigmoidSchedule,
    }

    if name not in schedules:
        raise ValueError(f"Unknown schedule '{name}'. Available: {list(schedules.keys())}")

    return schedules[name](**kwargs)
