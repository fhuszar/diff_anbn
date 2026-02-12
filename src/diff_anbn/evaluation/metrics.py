"""Evaluation metrics for generated samples."""

from dataclasses import dataclass

from diff_anbn.languages import FormalLanguage


@dataclass
class GenerationMetrics:
    """Metrics for a batch of generated samples."""

    syntactic_accuracy: float
    vocab_accuracy: float
    avg_length: float
    min_length: int
    max_length: int
    num_samples: int
    num_valid: int
    num_valid_vocab: int

    # Per-sample details
    samples: list[str]
    is_valid: list[bool]
    error_messages: list[str | None]


def compute_generation_metrics(
    samples: list[str],
    language: FormalLanguage,
) -> GenerationMetrics:
    """Compute metrics for generated samples.

    Args:
        samples: List of decoded strings
        language: Language to validate against

    Returns:
        GenerationMetrics with detailed results
    """
    num_samples = len(samples)

    if num_samples == 0:
        return GenerationMetrics(
            syntactic_accuracy=0.0,
            vocab_accuracy=0.0,
            avg_length=0.0,
            min_length=0,
            max_length=0,
            num_samples=0,
            num_valid=0,
            num_valid_vocab=0,
            samples=[],
            is_valid=[],
            error_messages=[],
        )

    vocab_set = set(language.vocab)
    is_valid = []
    error_messages = []
    valid_vocab = []
    lengths = []

    for s in samples:
        lengths.append(len(s))

        # Check vocabulary
        has_valid_vocab = all(c in vocab_set for c in s)
        valid_vocab.append(has_valid_vocab)

        # Check syntax
        result = language.validate_detailed(s)
        is_valid.append(result.is_valid)
        error_messages.append(result.error_message)

    num_valid = sum(is_valid)
    num_valid_vocab = sum(valid_vocab)

    return GenerationMetrics(
        syntactic_accuracy=num_valid / num_samples,
        vocab_accuracy=num_valid_vocab / num_samples,
        avg_length=sum(lengths) / num_samples,
        min_length=min(lengths),
        max_length=max(lengths),
        num_samples=num_samples,
        num_valid=num_valid,
        num_valid_vocab=num_valid_vocab,
        samples=samples,
        is_valid=is_valid,
        error_messages=error_messages,
    )


def compute_length_distribution(
    samples: list[str],
    language: FormalLanguage,
) -> dict[int, dict[str, int]]:
    """Compute validity by length.

    Args:
        samples: List of decoded strings
        language: Language to validate against

    Returns:
        Dictionary mapping length to {total, valid} counts
    """
    distribution: dict[int, dict[str, int]] = {}

    for s in samples:
        length = len(s)
        if length not in distribution:
            distribution[length] = {"total": 0, "valid": 0}

        distribution[length]["total"] += 1
        if language.validate(s):
            distribution[length]["valid"] += 1

    return distribution


def print_sample_analysis(
    samples: list[str],
    language: FormalLanguage,
    max_show: int = 10,
) -> None:
    """Print detailed analysis of samples.

    Args:
        samples: List of decoded strings
        language: Language to validate against
        max_show: Maximum samples to show per category
    """
    metrics = compute_generation_metrics(samples, language)

    print(f"\n{'='*60}")
    print(f"Generation Metrics")
    print(f"{'='*60}")
    print(f"Total samples:       {metrics.num_samples}")
    print(f"Syntactic accuracy:  {metrics.syntactic_accuracy:.2%}")
    print(f"Vocab accuracy:      {metrics.vocab_accuracy:.2%}")
    print(f"Average length:      {metrics.avg_length:.1f}")
    print(f"Length range:        [{metrics.min_length}, {metrics.max_length}]")

    # Show valid samples
    valid_samples = [s for s, v in zip(samples, metrics.is_valid) if v]
    print(f"\nValid samples ({len(valid_samples)} total):")
    for s in valid_samples[:max_show]:
        print(f"  {repr(s)}")
    if len(valid_samples) > max_show:
        print(f"  ... and {len(valid_samples) - max_show} more")

    # Show invalid samples with errors
    invalid_with_errors = [
        (s, e) for s, v, e in zip(samples, metrics.is_valid, metrics.error_messages)
        if not v
    ]
    print(f"\nInvalid samples ({len(invalid_with_errors)} total):")
    for s, e in invalid_with_errors[:max_show]:
        print(f"  {repr(s)}: {e}")
    if len(invalid_with_errors) > max_show:
        print(f"  ... and {len(invalid_with_errors) - max_show} more")
