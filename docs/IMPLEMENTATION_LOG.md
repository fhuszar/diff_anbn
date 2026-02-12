# Implementation Log

## Overview

This document tracks the implementation progress of the MDLM (Masked Diffusion Language Model) for toy formal languages.

---

## Phase 1: Project Setup

**Status**: Complete

- Created project structure with src layout
- Added pyproject.toml with dependencies
- Created package __init__.py files

---

## Phase 2: Language Module

**Status**: In Progress

### FormalLanguage Base Class
- Abstract interface for formal languages
- Methods: `generate()`, `validate()`, `validate_detailed()`, `vocab`

### AnBnLanguage
- Generates strings of form a^n b^n
- Validates by checking equal counts and correct ordering

### DyckLanguage
- Generates balanced parentheses
- Validates by stack-based parsing

### Tokenizer
- Character-level tokenization
- Special tokens: PAD, MASK, BOS, EOS

---

## Phase 3: Model Architecture

**Status**: Pending

### Design
- Using x-transformers with vanilla encoder config
- d_model=128, n_layers=4, n_heads=4
- Time conditioning via sinusoidal embeddings added to token embeddings

---

## Phase 4: MDLM Diffusion

**Status**: Pending

### Training
- Sample t ~ U(0,1)
- Mask tokens with probability (1 - alpha(t))
- Cross-entropy loss on masked positions

### Sampling
- Start fully masked
- Progressively unmask based on model predictions

---

## Phase 5: Training Infrastructure

**Status**: Pending

---

## Phase 6: Evaluation

**Status**: Pending

---

## Phase 7: Experiments

**Status**: Pending

---

## Notes

- MDLM chosen for simplicity (equivalent to weighted MLM)
- Starting with a^n b^n as the simplest context-free language
- Target: 100% syntactic accuracy on generations
