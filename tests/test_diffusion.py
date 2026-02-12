"""Tests for MDLM diffusion module."""

import pytest
import torch

from diff_anbn.diffusion.noise_schedule import (
    LinearSchedule,
    CosineSchedule,
    SigmoidSchedule,
    get_schedule,
)
from diff_anbn.diffusion.mdlm import MDLM
from diff_anbn.models.transformer import DiffusionTransformer


class TestNoiseSchedules:
    """Tests for noise schedules."""

    def test_linear_schedule_bounds(self):
        schedule = LinearSchedule()

        t = torch.tensor([0.0, 0.5, 1.0])
        alpha = schedule.alpha(t)

        assert torch.allclose(alpha, t)
        assert alpha[0] == 0.0
        assert alpha[2] == 1.0

    def test_cosine_schedule_bounds(self):
        schedule = CosineSchedule()

        # Check boundary conditions
        t_0 = torch.tensor([0.0])
        t_1 = torch.tensor([1.0])

        assert schedule.alpha(t_0).item() == pytest.approx(0.0, abs=1e-3)
        assert schedule.alpha(t_1).item() == pytest.approx(1.0, abs=1e-3)

    def test_cosine_schedule_monotonic(self):
        schedule = CosineSchedule()

        t = torch.linspace(0, 1, 100)
        alpha = schedule.alpha(t)

        # Should be monotonically increasing
        diffs = alpha[1:] - alpha[:-1]
        assert (diffs >= 0).all()

    def test_sigmoid_schedule_bounds(self):
        schedule = SigmoidSchedule()

        t_0 = torch.tensor([0.0])
        t_1 = torch.tensor([1.0])

        assert schedule.alpha(t_0).item() == pytest.approx(0.0, abs=1e-3)
        assert schedule.alpha(t_1).item() == pytest.approx(1.0, abs=1e-3)

    def test_mask_prob(self):
        schedule = LinearSchedule()

        t = torch.tensor([0.0, 0.5, 1.0])
        mask_prob = schedule.mask_prob(t)

        # mask_prob = 1 - alpha
        assert mask_prob[0] == 1.0  # Fully masked at t=0
        assert mask_prob[1] == 0.5
        assert mask_prob[2] == 0.0  # No masking at t=1

    def test_get_schedule(self):
        linear = get_schedule("linear")
        assert isinstance(linear, LinearSchedule)

        cosine = get_schedule("cosine")
        assert isinstance(cosine, CosineSchedule)

        with pytest.raises(ValueError):
            get_schedule("unknown")


class TestMDLM:
    """Tests for MDLM forward process and loss."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return DiffusionTransformer(
            vocab_size=6,  # PAD, MASK, BOS, EOS, a, b
            d_model=32,
            n_heads=2,
            n_layers=2,
            max_seq_len=16,
        )

    @pytest.fixture
    def mdlm(self, simple_model):
        """Create MDLM with simple model."""
        return MDLM(
            model=simple_model,
            noise_schedule=LinearSchedule(),
            mask_token_id=1,
            pad_token_id=0,
        )

    def test_forward_process_at_t0(self, mdlm):
        """At t=0, all tokens should be masked."""
        x = torch.tensor([[4, 4, 5, 5]])  # a a b b
        t = torch.tensor([0.0])

        x_masked, mask = mdlm.forward_process(x, t)

        assert mask.all()  # All should be masked
        assert (x_masked == mdlm.mask_token_id).all()

    def test_forward_process_at_t1(self, mdlm):
        """At t=1, no tokens should be masked."""
        x = torch.tensor([[4, 4, 5, 5]])  # a a b b
        t = torch.tensor([1.0])

        x_masked, mask = mdlm.forward_process(x, t)

        assert not mask.any()  # None should be masked
        assert (x_masked == x).all()

    def test_forward_process_intermediate(self, mdlm):
        """At intermediate t, some tokens should be masked."""
        torch.manual_seed(42)

        x = torch.tensor([[4, 4, 5, 5]] * 100)  # batch of 100
        t = torch.full((100,), 0.5)

        x_masked, mask = mdlm.forward_process(x, t)

        # With linear schedule at t=0.5, mask_prob = 0.5
        # So roughly 50% should be masked
        mask_rate = mask.float().mean().item()
        assert 0.3 < mask_rate < 0.7

    def test_compute_loss(self, mdlm):
        """Test loss computation."""
        x = torch.tensor([[4, 4, 5, 5], [4, 5, 5, 5]])  # batch of 2

        result = mdlm.compute_loss(x)

        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"].shape == ()  # scalar
        assert result["loss"].item() >= 0

    def test_get_logits(self, mdlm):
        """Test getting model logits."""
        x = torch.tensor([[4, 4, 5, 5]])
        t = torch.tensor([0.5])

        logits = mdlm.get_logits(x, t)

        assert logits.shape == (1, 4, 6)  # (batch, seq, vocab)
