"""Main training loop."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from diff_anbn.config import ExperimentConfig
from diff_anbn.diffusion import MDLM, sample
from diff_anbn.diffusion.noise_schedule import get_schedule
from diff_anbn.languages import Tokenizer, get_language
from diff_anbn.models import DiffusionTransformer

from .data import create_dataloader


@dataclass
class TrainingState:
    """Training state for checkpointing."""

    step: int = 0
    best_accuracy: float = 0.0
    metrics_history: list[dict] = field(default_factory=list)


class Trainer:
    """Main trainer class for MDLM."""

    def __init__(self, config: ExperimentConfig):
        """Initialize trainer.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.get_device())

        # Set random seeds
        self._set_seed(config.training.seed)

        # Setup output directories
        self.output_dir = config.output_dir / config.name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        for d in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(self.output_dir / "config.yaml")

        # Initialize components
        self.language = get_language(config.language.name)
        self.tokenizer = Tokenizer(vocab=self.language.vocab)

        self.model = DiffusionTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            max_seq_len=config.model.max_seq_len,
        ).to(self.device)

        noise_schedule = get_schedule(config.diffusion.noise_schedule)
        self.mdlm = MDLM(
            model=self.model,
            noise_schedule=noise_schedule,
            mask_token_id=self.tokenizer.mask_id,
            pad_token_id=self.tokenizer.pad_id,
        )

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = self._create_scheduler()

        # Create dataloader
        self.dataloader = create_dataloader(
            language=self.language,
            tokenizer=self.tokenizer,
            batch_size=config.training.batch_size,
            num_samples=config.training.batch_size * 100,  # Regenerate every 100 batches
            min_n=config.language.min_n,
            max_n=config.language.max_n,
            max_seq_len=config.model.max_seq_len,
        )

        # Training state
        self.state = TrainingState()

        # Log file
        self.log_file = open(self.log_dir / "metrics.jsonl", "a")

        # Initialize wandb if enabled
        self.wandb_run = None
        if config.wandb.enabled:
            import wandb

            self.wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.name,
                config=config.model_dump(mode="json"),
                tags=config.wandb.tags,
            )

        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Device: {self.device}")

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config.training.warmup_steps
        total_steps = self.config.training.num_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            # Linear decay after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, 1.0 - progress * 0.9)

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        """Run training loop."""
        config = self.config.training
        data_iter = iter(self.dataloader)

        pbar = tqdm(
            range(self.state.step, config.num_steps),
            initial=self.state.step,
            total=config.num_steps,
            desc="Training",
        )

        for step in pbar:
            self.state.step = step

            # Get batch (regenerate data iterator if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Training step
            metrics = self._train_step(batch)

            # Update progress bar
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.3f}")

            # Logging
            if step % config.log_every == 0:
                self._log_metrics(step, metrics)

            # Evaluation
            if step > 0 and step % config.eval_every == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(step, eval_metrics, prefix="eval")

                if eval_metrics["syntactic_accuracy"] > self.state.best_accuracy:
                    self.state.best_accuracy = eval_metrics["syntactic_accuracy"]
                    self._save_checkpoint("best")

            # Checkpointing
            if step > 0 and step % config.save_every == 0:
                self._save_checkpoint(f"step_{step}")

        # Final evaluation and checkpoint
        eval_metrics = self.evaluate()
        self._log_metrics(self.state.step, eval_metrics, prefix="eval")
        self._save_checkpoint("final")

        self.log_file.close()

        # Finish wandb run
        if self.wandb_run:
            self.wandb_run.summary["best_accuracy"] = self.state.best_accuracy
            self.wandb_run.finish()

        print(f"\nTraining complete. Best accuracy: {self.state.best_accuracy:.4f}")

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pass
        result = self.mdlm.compute_loss(input_ids, attention_mask)
        loss = result["loss"]

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": float(loss.item()),
            "accuracy": float(result["accuracy"].item()),
            "mask_rate": float(result["mask_rate"].item()),
            "lr": float(self.scheduler.get_last_lr()[0]),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model by generating samples and checking validity."""
        self.model.eval()
        config = self.config

        # Generate samples
        num_samples = config.training.num_eval_samples
        seq_len = config.model.max_seq_len

        samples = sample(
            mdlm=self.mdlm,
            batch_size=num_samples,
            seq_len=seq_len,
            num_steps=config.diffusion.num_timesteps,
            temperature=1.0,
            device=self.device,
            show_progress=False,
        )

        # Decode and validate
        decoded = self.tokenizer.decode_batch(samples, skip_special=True)

        valid_count = sum(1 for s in decoded if self.language.validate(s))
        syntactic_accuracy = valid_count / num_samples

        # Check vocabulary validity (all chars in vocab)
        vocab_set = set(self.language.vocab)
        valid_vocab_count = sum(
            1 for s in decoded if all(c in vocab_set for c in s)
        )
        vocab_accuracy = valid_vocab_count / num_samples

        # Average length
        avg_length = sum(len(s) for s in decoded) / num_samples

        return {
            "syntactic_accuracy": syntactic_accuracy,
            "vocab_accuracy": vocab_accuracy,
            "avg_length": avg_length,
            "num_samples": num_samples,
        }

    def _log_metrics(
        self, step: int, metrics: dict[str, float], prefix: str = "train"
    ) -> None:
        """Log metrics to file and wandb."""
        log_entry = {
            "step": step,
            "prefix": prefix,
            **{f"{prefix}/{k}": v for k, v in metrics.items()},
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()

        self.state.metrics_history.append(log_entry)

        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=step,
            )

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "step": self.state.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.state.best_accuracy,
            "config": self.config.model_dump(),
        }
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)

        # Save best model as wandb artifact
        if self.wandb_run and name == "best":
            import wandb

            artifact = wandb.Artifact(f"{self.config.name}-best", type="model")
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.state.step = checkpoint["step"]
        self.state.best_accuracy = checkpoint["best_accuracy"]

        print(f"Loaded checkpoint from step {self.state.step}")

    def generate_samples(
        self, num_samples: int = 10, temperature: float = 1.0
    ) -> list[str]:
        """Generate and return decoded samples."""
        self.model.eval()

        samples = sample(
            mdlm=self.mdlm,
            batch_size=num_samples,
            seq_len=self.config.model.max_seq_len,
            num_steps=self.config.diffusion.num_timesteps,
            temperature=temperature,
            device=self.device,
            show_progress=False,
        )

        return self.tokenizer.decode_batch(samples, skip_special=True)
