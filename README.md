# PIONEER: Platform for Iterative Optimization and Navigation to Enhance Exploration of Regulatory sequences

PIONEER is a PyTorch framework for iterative sequence optimization through active learning and guided sequence generation.


## Installation

```bash
pip install pioneer-nn
```

## Core Components

### 1. Sequence Generation (`generator.py`)
- `RandomGenerator`: Creates sequences based on nucleotide probabilities
- `MutagenesisGenerator`: Random mutations within specified windows
- `GuidedMutagenesisGenerator`: Attribution-guided mutations within specified windows
- `PoolBasedGenerator`: Selects sequences from a predefined pool based on a scoring function
- `SequentialProposer`: Applies multiple generators in sequence
- `MultiProposer`: Applies generators in parallel

### 2. Sequence Selection (`acquisition.py`)
- `RandomAcquisition`: Random sequence sampling
- `ScoreAcquisition`: Selection based on arbitrary scoring function
- `LCMDAcquisition`: Linear Centered Maximum Distance selection

### 3. Attribution Methods (`attribution.py`)
- `Saliency`: Gradient-based attribution scores for input sequences based on a scorer function

### 4. Prediction Methods (`predictor.py`)
- `Scalar`: For models outputting scalar values
- `Profile`: For models outputting position-wise profiles
- Supports multi-task outputs through task indexing

### 5. Uncertainty Estimation (`uncertainty.py`)
- `MCDropout`: Monte Carlo Dropout sampling
- `DeepEnsemble`: Ensemble-based uncertainty

### 6. Oracle Interface (`oracle.py`)
- `SingleOracle`: Single model predictions
- `EnsembleOracle`: Ensemble model predictions with uncertainty

### 7. Model Wrapping (`surrogate.py`)
- `ModelWrapper`: Unified interface for predictions and uncertainty

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
        predictor=predictor.Scalar()
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
new_seqs, new_labels = pioneer.run_cycle(
    x=train_seqs,
    y=train_labels,
    val_x=val_seqs,
    val_y=val_labels,
    trainer_factory=lambda: pl.Trainer(max_epochs=10)
)
```

## Advanced Usage

### Sequential Generation
```python
from pioneer import generator

gen = generator.SequentialProposer([
    generator.RandomGenerator(prob=[0.3, 0.2, 0.2, 0.3]),
    generator.MutagenesisGenerator(mut_rate=0.1)
])
```

### Ensemble Predictions
```python
from pioneer import oracle, predictor

oracle_model = oracle.EnsembleOracle(
    model_class=YourModel,
    model_kwargs={'hidden_dim': 256},
    weight_paths=['model1.pt', 'model2.pt', 'model3.pt'],
    predictor=predictor.Scalar()
)
```

### Multi-Task Predictions
```python
from pioneer import predictor

# For scalar outputs
pred = predictor.Scalar(task_index=0)  # Select first task

# For profile outputs
pred = predictor.Profile(
    reduction=torch.mean,  # Custom reduction
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
- PyTorch Lightning

## Citation

```bibtex
@article{pioneer2024,
  title={PIONEER: An in silico playground for iterative improvement of genomic deep learning},
  author={A Crnjar, J Desmarais, JB Kinney, PK Koo},
  journal={bioRxiv},
  year={2024}
}
```

