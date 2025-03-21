# PIONEER: Platform for Iterative Optimization and Navigation to Enhance Exploration of Regulatory sequences

PIONEER is a PyTorch framework for iterative sequence optimization through active learning and guided sequence generation.


## Installation

```bash
pip install pioneer-nn
```

## Core Components

### 1. Sequence Proposers (`proposer.py`)
Base class `Proposer` with implementations:
- `SequentialProposer`: Applies multiple proposers in sequence
- `MultiProposer`: Applies proposers in parallel and combines results

### 2. Sequence Generation (`generator.py`)
Base class `Generator` with implementations:
- `RandomGenerator`: Creates sequences based on nucleotide probabilities
- `MutagenesisGenerator`: Random mutations within specified windows
- `GuidedMutagenesisGenerator`: Attribution-guided mutations
- `PoolBasedGenerator`: Selects sequences from a predefined pool

### 3. Sequence Selection (`acquisition.py`)
Base class `Acquisition` with implementations:
- `RandomAcquisition`: Random sequence sampling
- `ScoreAcquisition`: Selection based on arbitrary scoring function
- `LCMDAcquisition`: Linear Centered Maximum Distance selection

### 4. Attribution Methods (`attribution.py`)
Base class `AttributionMethod` with implementations:
- `Saliency`: Gradient-based attribution scores for input sequences
- Supports both prediction-based and uncertainty-based attribution

### 5. Prediction Methods (`predictor.py`)
Base class `Predictor` with implementations:
- `Scalar`: For models outputting scalar values
- `Profile`: For models outputting position-wise profiles with custom reductions

### 6. Uncertainty Estimation (`uncertainty.py`)
Base class `UncertaintyMethod` with implementations:
- `MCDropout`: Monte Carlo Dropout sampling
- `DeepEnsemble`: Ensemble-based uncertainty

### 7. Oracle Interface (`oracle.py`)
Base class `Oracle` with implementations:
- `SingleOracle`: Single model predictions
- `EnsembleOracle`: Ensemble predictions with uncertainty estimation

### 8. Model Wrapping (`surrogate.py`)
- `ModelWrapper`: Unified interface for predictions and uncertainty
- Supports both PyTorch and PyTorch Lightning models

### 9. Utilities (`utils.py`)
- `upsample`: Function to increase dataset size by repeating sequences

## Quick Start

```python
from pioneer import (
    PIONEER,
    generator, 
    acquisition,
    uncertainty,
    oracle,
    predictor,
    surrogate
)

# Setup components
model = YourModel()
predictor = predictor.Scalar(task_index=0)  # For multi-task models
uncertainty = uncertainty.MCDropout(n_samples=20)
wrapper = surrogate.ModelWrapper(model, predictor, uncertainty)

# Initialize PIONEER
pioneer = PIONEER(
    model=wrapper,
    oracle=oracle.SingleOracle(
        model_class=OracleModel,
        model_kwargs={'hidden_dim': 256},
        weight_path='oracle_weights.pt',
        predictor=predictor.Scalar(),
        model_type='pytorch'  # or 'lightning'
    ),
    generator=generator.MutagenesisGenerator(mut_rate=0.1),
    acquisition=acquisition.ScoreAcquisition(
        target_size=1000,
        scorer=wrapper.uncertainty
    ),
    batch_size=32,
    cold_start=True
)

# Run optimization cycle
x_new, y_new = pioneer.run_cycle(
    x=train_seqs,
    y=train_labels,
    trainer_factory=lambda: pl.Trainer(max_epochs=10)
)
```

## Advanced Usage

### Sequential and Parallel Proposers
```python
from pioneer import proposer, generator

# Sequential application
seq_gen = proposer.SequentialProposer([
    generator.RandomGenerator(prob=[0.3, 0.2, 0.2, 0.3]),
    generator.MutagenesisGenerator(mut_rate=0.1)
])

# Parallel application
multi_gen = proposer.MultiProposer([
    generator.RandomGenerator(prob=[0.25, 0.25, 0.25, 0.25]),
    generator.GuidedMutagenesisGenerator(attr_method, mut_rate=0.2)
])
```

### Ensemble Predictions with Uncertainty
```python
from pioneer import oracle, predictor

oracle_model = oracle.EnsembleOracle(
    model_class=YourModel,
    model_kwargs={'hidden_dim': 256},
    weight_paths=['model1.pt', 'model2.pt', 'model3.pt'],
    predictor=predictor.Scalar(),
    model_type='pytorch'
)

# Get predictions and uncertainties
preds, uncert = oracle_model.predict_uncertainty(sequences)
```

### Profile Predictions with Custom Reduction
```python
from pioneer import predictor
import torch

# Profile predictor with max reduction
profile_pred = predictor.Profile(
    reduction=lambda x, dim: torch.max(x, dim=dim)[0],
    task_index=0
)
```

## Input/Output Formats

- Sequences: `(N, A, L)` tensors where:
  - N: Batch size
  - A: Alphabet size (typically 4 for DNA)
  - L: Sequence length
- Labels: Task-dependent shapes
  - Scalar: `(N,)` or `(N, T)` for T tasks
  - Profile: `(N, L)` or `(N, T, L)` for T tasks

## Dependencies

- PyTorch ≥ 1.12
- PyTorch Lightning
- NumPy
- SciPy (for correlation metrics)

## Citation

```bibtex
@article{pioneer2024,
  title={PIONEER: An in silico playground for iterative improvement of genomic deep learning},
  author={A Crnjar, J Desmarais, JB Kinney, PK Koo},
  journal={bioRxiv},
  year={2024}
}
```

