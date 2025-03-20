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
    uncertainties = uncertainty(model, sample_data)
    
    # Test output shape
    assert uncertainties.shape == (N,), "Uncertainty shape mismatch"
    assert uncertainties.dtype == torch.float32, "Uncertainty dtype mismatch"
    
    # Test that uncertainties are non-negative
    assert torch.all(uncertainties >= 0), "Negative uncertainty values found"
    
    # Test that higher dropout rates lead to higher average uncertainty
    if N > 1:
        model_low = MockDropoutModel(A=A, L=L, dropout_rate=0.1).to(device)
        model_high = MockDropoutModel(A=A, L=L, dropout_rate=0.9).to(device)
        
        uncertainties_low = uncertainty(model_low, sample_data)
        uncertainties_high = uncertainty(model_high, sample_data)
        
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
    uncertainties = uncertainty(models.models, sample_data)
    
    # Test output shape
    assert uncertainties.shape == (N,), "Uncertainty shape mismatch"
    assert uncertainties.dtype == torch.float32, "Uncertainty dtype mismatch"
    
    # Test that uncertainties are non-negative
    assert torch.all(uncertainties >= 0), "Negative uncertainty values found"
    
    # Test that higher model variance leads to higher uncertainty
    if N > 1:
        models_low = MockEnsembleModel(A=A, L=L, n_models=n_models, variance=0.01).to(device)
        models_high = MockEnsembleModel(A=A, L=L, n_models=n_models, variance=1.0).to(device)
        
        uncertainties_low = uncertainty(models_low.models, sample_data)
        uncertainties_high = uncertainty(models_high.models, sample_data)
        
        assert uncertainties_low.mean() < uncertainties_high.mean(), "Higher model variance should lead to higher uncertainty"

