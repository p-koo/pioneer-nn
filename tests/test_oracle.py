import torch
import pytest
import sys
sys.path.append('./')
from pioneer.oracle import SingleOracle, EnsembleOracle
from pioneer.predictor import Scalar
import pytorch_lightning as pl
@pytest.fixture(params=[
    (10, 4, 100),  # original size
    (10, 4, 50),   # shorter sequence
    (10, 4, 200),  # longer sequence
    (32, 4, 100),  # one batch
    (46, 4, 100),  # 1.5 batches
    (64, 4, 100),  # two batches
    (1, 4, 1)      # single position
])
def sample_data(request):
    N, A, L = request.param  # batch size, alphabet size, sequence length
    x = torch.zeros(N*L,A,1)
    nt_idx = torch.randint(0,A,(N*L,))
    x[torch.arange(N*L),nt_idx] = 1  # One-hot encode random nucleotide
    x = x.view(N,L,A)
    x = x.transpose(1,2)

    # Test one-hot property
    assert (x.sum(dim=1) == 1).all(), "One-hot property mismatch"
    # Check that all elements are either 0 or 1
    assert torch.all(torch.logical_or(x == 0, x == 1)), "Elements are not 0 or 1"
    return x

class MockModel(torch.nn.Module):
    """Mock model for testing oracles"""
    def __init__(self, A, L, n_tasks=1):
        super().__init__()
        self.A = A
        self.L = L
        self.n_tasks = n_tasks
        self.linear = torch.nn.Linear(A*L, n_tasks)
        # Initialize weights for predictable behavior
        torch.nn.init.ones_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
    
class MockLightningModel(pl.LightningModule):
    def __init__(self, A, L, n_tasks=1):
        super().__init__()
        self.A = A
        self.L = L
        self.n_tasks = n_tasks
        self.linear = torch.nn.Linear(A*L, n_tasks)
        # Initialize weights for predictable behavior
        torch.nn.init.ones_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch

@pytest.mark.parametrize("n_tasks", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 32, 64])
@pytest.mark.parametrize("model_type", ['pytorch', 'lightning'])
def test_single_oracle(sample_data, n_tasks, batch_size, tmp_path, model_type):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create temporary weight file
    weight_path = tmp_path / "model_weights.pt"
    if model_type == 'pytorch':
        model_class = MockModel
        model = model_class(A=A, L=L, n_tasks=n_tasks)
        torch.save(model.state_dict(), weight_path)
    elif model_type == 'lightning':
        model_class = MockLightningModel
        model = model_class(A=A, L=L, n_tasks=n_tasks)
        trainer = pl.Trainer(max_epochs=1)
        trainer.strategy.connect(model)
        trainer.save_checkpoint(weight_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
    
    # Initialize oracle
    predictor = Scalar(task_index=0 if n_tasks > 1 else None)
    oracle = SingleOracle(
        model_class=model_class,
        model_kwargs={'A': A, 'L': L, 'n_tasks': n_tasks},
        weight_path=weight_path,
        predictor=predictor,
        device=device,
        model_type=model_type
    )
    
    # Test predictions
    predictions = oracle.predict(sample_data, batch_size)
    
    # Test output shape
    assert predictions.shape == (N, 1), "Prediction shape mismatch"
    assert predictions.dtype == torch.float32, "Prediction dtype mismatch"
    
    # Test predictions match expected values (sum of inputs due to weight initialization)
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(predictions, expected), "Predictions don't match expected values"
    

@pytest.mark.parametrize("n_models", [3, 5])
@pytest.mark.parametrize("n_tasks", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 32, 64])
@pytest.mark.parametrize("model_type", ['pytorch', 'lightning'])
def test_ensemble_oracle(sample_data, n_models, n_tasks, batch_size, tmp_path, model_type):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create temporary weight files
    weight_paths = []
    for i in range(n_models):
        weight_path = tmp_path / f"model_{i}_weights.pt"
        if model_type == 'pytorch':
            model_class = MockModel
            model = model_class(A=A, L=L, n_tasks=n_tasks)
            # Add small variance to weights for each model
            model.linear.weight.data += torch.randn_like(model.linear.weight.data) * 0.1
            torch.save(model.state_dict(), weight_path)
        elif model_type == 'lightning':
            model_class = MockLightningModel
            model = model_class(A=A, L=L, n_tasks=n_tasks)
            model.linear.weight.data += torch.randn_like(model.linear.weight.data) * 0.1
            trainer = pl.Trainer(max_epochs=1)
            trainer.strategy.connect(model)
            trainer.save_checkpoint(weight_path)
        else:
            raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
        weight_paths.append(weight_path)
    
    # Initialize oracle
    predictor = Scalar(task_index=0 if n_tasks > 1 else None)
    oracle = EnsembleOracle(
        model_class=model_class,
        model_kwargs={'A': A, 'L': L, 'n_tasks': n_tasks},
        weight_paths=weight_paths,
        predictor=predictor,
        device=device,
        model_type=model_type
    )
    
    # Test predictions
    predictions = oracle.predict(sample_data, batch_size)
    
    # Test output shape
    assert predictions.shape == (N, 1), "Prediction shape mismatch"
    assert predictions.dtype == torch.float32, "Prediction dtype mismatch"
    
    # Test predictions are close to but not exactly equal to sum of inputs
    # (due to weight variations in ensemble members)
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(predictions, expected, rtol=0.2), "Predictions too far from expected values"
    assert not torch.allclose(predictions, expected, rtol=1e-5), "Predictions exactly match input sum"
    
    # Test batched predictions
    predictions_batched = []
    for i in range(0, N, batch_size):
        batch = sample_data[i:i+batch_size]
        predictions_batched.append(oracle.predict(batch, batch_size))
    predictions_batched = torch.cat(predictions_batched)
    
    assert torch.allclose(predictions, predictions_batched), "Batched predictions don't match"

@pytest.mark.parametrize("n_models", [3, 5])
@pytest.mark.parametrize("model_type", ['pytorch', 'lightning'])
def test_ensemble_variance(sample_data, n_models, model_type, tmp_path):
    """Test that ensemble predictions have higher variance with more diverse weights"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    def create_ensemble(variance):
        weight_paths = []
        for i in range(n_models):
            weight_path = tmp_path / f"model_{i}_weights_{variance}.pt"
            if model_type == 'pytorch':
                model_class = MockModel
                model = model_class(A=A, L=L)
                model.linear.weight.data += torch.randn_like(model.linear.weight.data) * variance
                torch.save(model.state_dict(), weight_path)
            elif model_type == 'lightning':
                model_class = MockLightningModel
                model = model_class(A=A, L=L)
                model.linear.weight.data += torch.randn_like(model.linear.weight.data) * variance
                trainer = pl.Trainer(max_epochs=1)
                trainer.strategy.connect(model)
                trainer.save_checkpoint(weight_path)
            else:
                raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
            weight_paths.append(weight_path)
        return weight_paths
    
    # Create two ensembles with different weight variances
    if model_type == 'pytorch':
        model_class = MockModel
    elif model_type == 'lightning':
        model_class = MockLightningModel
    else:
        raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
    low_var = 0.1
    high_var = 5
    predictor = Scalar()
    oracle_low_var = EnsembleOracle(
        model_class=model_class,
        model_kwargs={'A': A, 'L': L},
        weight_paths=create_ensemble(low_var),
        predictor=predictor,
        device=device,
        model_type=model_type
    )
    
    oracle_high_var = EnsembleOracle(
        model_class=model_class,
        model_kwargs={'A': A, 'L': L},
        weight_paths=create_ensemble(high_var),
        predictor=predictor,
        device=device,
        model_type=model_type
    )
    
    # Get predictions
    preds_low_var, uncertainty_low_var = oracle_low_var.predict_uncertainty(sample_data)
    preds_high_var, uncertainty_high_var = oracle_high_var.predict_uncertainty(sample_data)

    # Test shapes
    assert preds_low_var.shape == (N,1), "Prediction shapes mismatch"
    assert uncertainty_low_var.shape == (N,1), "Uncertainty shapes mismatch"
    assert preds_high_var.shape == (N,1), "Prediction shapes mismatch"
    assert uncertainty_high_var.shape == (N,1), "Uncertainty shapes mismatch"

    #Test data type
    assert preds_low_var.dtype == torch.float32, "Prediction dtype mismatch"
    assert uncertainty_low_var.dtype == torch.float32, "Uncertainty dtype mismatch"
    assert preds_high_var.dtype == torch.float32, "Prediction dtype mismatch"
    assert uncertainty_high_var.dtype == torch.float32, "Uncertainty dtype mismatch"

    #test that predictions are close to expected values
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(preds_low_var, expected, rtol=2*low_var), "Predictions too far from expected values"
    assert torch.allclose(preds_high_var, expected, rtol=2*high_var), "Predictions too far from expected values"

    # Test that higher weight variance leads to higher prediction variance
    comparison = torch.lt(uncertainty_low_var.flatten(), uncertainty_high_var.flatten())
    assert comparison.sum() > len(comparison)//2, "Higher weight variance should lead to higher prediction variance"