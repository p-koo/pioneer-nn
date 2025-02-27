# PIONEER: Platform for Iterative Optimization and Navigation to Enhance Exploration of Regulatory sequences

PIONEER is a PyTorch framework for iterative sequence optimization through active learning and guided sequence generation.

## Core Components

### 1. Sequence Generation (`generator.py`)
- `generator.Random`: Creates sequences based on nucleotide probabilities
- `generator.Mutagenesis`: Random mutations within specified windows
- `generator.GuidedMutagenesis`: Attribution-guided mutations
- `generator.Sequential`: Applies multiple generators in sequence
- `generator.MultiGenerator`: Applies generators in parallel

### 2. Sequence Selection (`acquisition.py`)
- `acquisition.Random`: Random sequence sampling
- `acquisition.Uncertainty`: Selection based on model uncertainty
- `acquisition.LCMD`: Low-Confidence-Maximum-Distance selection

### 3. Attribution Methods (`attribution.py`)
- `attribution.UncertaintySaliency`: Gradients of uncertainty w.r.t inputs
- `attribution.ActivitySaliency`: Gradients of predictions w.r.t inputs

### 4. Prediction Methods (`predictor.py`)
- `predictor.Scalar`: For models outputting scalar values
- `predictor.Profile`: For models outputting position-wise profiles
- Supports multi-task outputs through task indexing

### 5. Uncertainty Estimation (`uncertainty.py`)
- `uncertainty.MCDropout`: Monte Carlo Dropout sampling
- `uncertainty.DeepEnsemble`: Ensemble-based uncertainty

### 6. Oracle Interface (`oracle.py`)
- `oracle.OracleSingle`: Single model predictions
- `oracle.OracleEnsemble`: Ensemble model predictions with uncertainty

### 7. Model Wrapping (`surrogate.py`)
- `surrogate.ModelWrapper`: Unified interface for predictions and uncertainty

## Quick Start

```python
from pioneer import (
    pioneer,
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
pioneer = pioneer.PIONEER(
    model=wrapper,
    oracle=oracle.OracleSingle(oracle_model),
    generator=generator.Mutagenesis(mut_rate=0.1),
    selector=acquisition.Uncertainty(target_size=1000, surrogate_model=wrapper)
)

# Run optimization cycle
new_seqs, new_labels = pioneer.run_cycle(
    x=train_seqs,
    y=train_labels,
    val_x=val_seqs,
    val_y=val_labels
)
```

## Advanced Usage

### Guided Sequence Generation
```python
from pioneer import generator, attribution

attr = attribution.UncertaintySaliency(model)
gen = generator.GuidedMutagenesis(
    attr_method=attr,
    mut_rate=0.1,
    mut_window=(10, 20),
    temp=1.0
)
```

### Ensemble Predictions
```python
from pioneer import oracle

oracle_model = oracle.OracleEnsemble(
    model=base_model,
    weight_paths=['model1.h5', 'model2.h5', 'model3.h5']
)
```

### Multi-Task Predictions
```python
from pioneer import predictor

# For scalar outputs
pred = predictor.Scalar(task_index=0)  # Select first task

# For profile outputs
pred = predictor.Profile(
    reduction=lambda x: x.mean(dim=-1),  # Custom reduction
    task_index=1  # Select second task
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

## Citation

```bibtex
@article{pioneer2024,
  title={PIONEER: An in silico playground for iterative improvement of genomic deep learning},
  author={A Crjnar, J Desmarais, JB Kinney, PK Koo},
  journal={bioRxiv},
  year={2024}
}
```

