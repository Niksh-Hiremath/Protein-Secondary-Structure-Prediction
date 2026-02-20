# Protein Secondary Structure Prediction (Q8/Q3)

PyTorch implementation of bidirectional sequence-to-sequence models for token-level protein secondary structure prediction under Q8 and Q3 annotation schemes.

---

## Overview

Predicting protein secondary structure is a fundamental problem in computational biology. Given an amino acid sequence, the goal is to predict the structural label for each residue.

This project formulates the task as a sequence-to-sequence tagging problem:

- **Input**: Amino acid sequence (e.g., "ACDEFGHIK")
- **Outputs**:
  - Q8: Eight-state secondary structure labels
  - Q3: Three-state secondary structure labels (derived from Q8)

Both outputs are predicted at the residue level and must match the input sequence length.

---

## Problem Formulation

Each protein sequence is treated as a variable-length token sequence.

- Tokenization over standard amino acids (+ masked `*`)
- Learned embedding representation
- Bidirectional recurrent modeling
- Multi-task output heads for Q8 and Q3
- Token-level classification using CrossEntropyLoss
- Evaluation using token-level F1 score

Final score: Harmonic mean of F1(Q8) and F1(Q3)

---

## Dataset

### Training Data (`train.csv`)
- `id` — Sequence identifier
- `seq` — Amino acid sequence
- `sst8` — Q8 secondary structure
- `sst3` — Q3 secondary structure

Predictions must:
- Match input sequence length
- Provide both Q8 and Q3 outputs

Q3 is derived from Q8:

- Helix: (H, G, I) → H
- Strand: (E, B) → E
- Coil: (C, S, T) → C

---

## Model Architectures

Two scratch models were implemented:

### 1. Bidirectional RNN
- Embedding layer
- Bidirectional RNN
- Two linear heads (Q8, Q3)

### 2. Bidirectional LSTM
- Embedding layer
- Stacked Bidirectional LSTM
- Two linear heads (Q8, Q3)

Both models support:
- Variable-length sequences
- Padding and masking
- Multi-task loss optimization

---

## Training Setup

- Framework: PyTorch
- Optimizer: Adam
- Loss: CrossEntropyLoss (Q8 + Q3)
- Evaluation: Token-level F1 score
- Experiment tracking: TrackIO
- Model versioning: KaggleHub

---

## Training

Training code defines:

- Dataset and DataLoader
- Model initialization
- Loss computation
- Training and validation loops
- Metric logging
- Model checkpoint saving

---

## Inference

The inference pipeline:

1. Load trained model weights from KaggleHub
2. Tokenize test sequences
3. Perform forward pass
4. Decode predictions
5. Generate `submission.csv`

Predictions strictly preserve sequence length.

---

## Results

Both architectures were evaluated on token-level F1 score.

Bidirectional LSTM consistently outperformed vanilla RNN due to improved long-range dependency modeling.

---

## Key Features

- Scratch implementation (no pretrained models)
- Multi-task learning (Q8 + Q3)
- Padding-aware bidirectional modeling
- Modular and reproducible training pipeline

---
