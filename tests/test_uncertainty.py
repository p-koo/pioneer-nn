import torch
import pytest
import sys
sys.path.append('./pioneer')
from uncertainty import MCDropout, DeepEnsemble

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

class MockDropoutModel(torch.nn.Module):
    """Mock model with dropout for testing MC Dropout"""
    def __init__(self, A, L, dropout_rate=0.5):
        super().__init__()
        self.A = A
        self.L = L
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(A*L, 1)
        )
        
    def forward(self, x):
        return self.layers(x.flatten(start_dim=1))

@pytest.mark.parametrize("n_samples", [10, 20, 50])
@pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.9])
def test_mc_dropout(sample_data, n_samples, dropout_rate):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and uncertainty estimator
    model = MockDropoutModel(A=A, L=L, dropout_rate=dropout_rate).to(device)
    uncertainty = MCDropout(n_samples=n_samples)
    
    # Get uncertainty estimates
    uncertainties = uncertainty.estimate(model, sample_data)
    
    # Test output shape
    assert uncertainties.shape == (N,), "Uncertainty shape mismatch"
    assert uncertainties.dtype == torch.float32, "Uncertainty dtype mismatch"
    
    # Test that uncertainties are non-negative
    assert torch.all(uncertainties >= 0), "Negative uncertainty values found"
    
    # Test that higher dropout rates lead to higher average uncertainty
    if N > 1:
        model_low = MockDropoutModel(A=A, L=L, dropout_rate=0.1).to(device)
        model_high = MockDropoutModel(A=A, L=L, dropout_rate=0.9).to(device)
        
        uncertainties_low = uncertainty.estimate(model_low, sample_data)
        uncertainties_high = uncertainty.estimate(model_high, sample_data)
        
        assert uncertainties_low.mean() < uncertainties_high.mean(), "Higher dropout rate should lead to higher uncertainty"

class MockEnsembleModel(torch.nn.ModuleList):
    """Mock ensemble model for testing Deep Ensemble"""
    def __init__(self, A, L, n_models=5, variance=0.1):
        super().__init__()
        self.A = A
        self.L = L
        
        # Create ensemble members with different weight initializations
        self.models = torch.nn.ModuleList(MockDropoutModel(A=A, L=L, dropout_rate=0) for _ in range(n_models))
        for model in self.models:
            model.layers[1].weight.data = torch.randn_like(model.layers[1].weight.data) * variance + 1.0
            model.layers[1].bias.data.fill_(0.0)

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        return torch.stack([model(x_flat) for model in self.models]).mean(dim=0)

@pytest.mark.parametrize("n_models", [3, 5, 10])
@pytest.mark.parametrize("model_variance", [0.01, 0.1, 1.0])
def test_deep_ensemble(sample_data, n_models, model_variance):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize ensemble and uncertainty estimator
    models = MockEnsembleModel(A=A, L=L, n_models=n_models, variance=model_variance).to(device)
    uncertainty = DeepEnsemble()
    
    # Get uncertainty estimates
    uncertainties = uncertainty.estimate(models, sample_data)
    
    # Test output shape
    assert uncertainties.shape == (N,), "Uncertainty shape mismatch"
    assert uncertainties.dtype == torch.float32, "Uncertainty dtype mismatch"
    
    # Test that uncertainties are non-negative
    assert torch.all(uncertainties >= 0), "Negative uncertainty values found"
    
    # Test that higher model variance leads to higher uncertainty
    if N > 1:
        models_low = MockEnsembleModel(A=A, L=L, n_models=n_models, variance=0.01).to(device)
        models_high = MockEnsembleModel(A=A, L=L, n_models=n_models, variance=1.0).to(device)
        
        uncertainties_low = uncertainty.estimate(models_low, sample_data)
        uncertainties_high = uncertainty.estimate(models_high, sample_data)
        
        assert uncertainties_low.mean() < uncertainties_high.mean(), "Higher model variance should lead to higher uncertainty"

# @pytest.mark.parametrize("batch_size", [1, 32, 64])
# def test_uncertainty_batching(sample_data, batch_size):
#     N, A, L = sample_data.shape
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Test both uncertainty methods
#     dropout_model = MockDropoutModel(A=A, L=L).to(device)
#     ensemble_models = MockEnsembleModel(A=A, L=L).to(device)
    
#     mc_dropout = MCDropout(n_samples=20)
#     deep_ensemble = DeepEnsemble()
    
#     # Get full uncertainties
#     uncertainties_dropout = mc_dropout.estimate(dropout_model, sample_data)
#     uncertainties_ensemble = deep_ensemble.estimate(ensemble_models, sample_data)
    
#     # Get batched uncertainties
#     uncertainties_dropout_batched = []
#     uncertainties_ensemble_batched = []
#     for i in range(0, N, batch_size):
#         batch = sample_data[i:i+batch_size]
#         uncertainties_dropout_batched.append(mc_dropout.estimate(dropout_model, batch))
#         uncertainties_ensemble_batched.append(deep_ensemble.estimate(ensemble_models, batch))
    
#     uncertainties_dropout_batched = torch.cat(uncertainties_dropout_batched)
#     uncertainties_ensemble_batched = torch.cat(uncertainties_ensemble_batched)
    
#     # Test batched results match full results (within numerical tolerance)
#     assert torch.allclose(uncertainties_dropout, uncertainties_dropout_batched, rtol=1e-4), "MC Dropout batched uncertainties don't match"
#     assert torch.allclose(uncertainties_ensemble, uncertainties_ensemble_batched, rtol=1e-4), "Deep Ensemble batched uncertainties don't match" 