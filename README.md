# Discrete Diffusion Language Models for Toy Formal Languages

Train toy-sized masked diffusion language models (MDLM) on formal languages and demonstrate that syntactic correctness of generations improves during training.

## Languages

- **a^n b^n**: Strings of n 'a's followed by n 'b's (context-free)
- **Dyck-1**: Balanced parentheses (context-free)

## Installation

```bash
# Using uv
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Usage

### Training

```bash
python scripts/train.py --config configs/experiment/anbn.yaml
```

### Generation

```bash
python scripts/generate.py --checkpoint outputs/checkpoints/model.pt --num_samples 100
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/model.pt
```

## Architecture

**MDLM (Masked Diffusion Language Model)**:
- Training: Sample t ~ U(0,1), mask tokens with probability (1-α(t)), predict original tokens
- Sampling: Start fully masked, progressively unmask based on model predictions

**Model**: Small transformer encoder (d_model=128, n_layers=4, n_heads=4)

## Project Structure

```
diff_anbn/
├── src/diff_anbn/
│   ├── languages/       # Formal language definitions
│   ├── models/          # Transformer architecture
│   ├── diffusion/       # MDLM implementation
│   ├── evaluation/      # Metrics and visualization
│   ├── training/        # Training loop
│   └── config/          # Pydantic schemas
├── configs/             # YAML experiment configs
├── scripts/             # Entry points
└── tests/               # Unit tests
```

## Results

*Coming soon: Syntactic accuracy curves showing improvement to 100%*

## License

MIT
