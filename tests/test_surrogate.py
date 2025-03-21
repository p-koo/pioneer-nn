import torch
import pytest
import sys
sys.path.append('./pioneer')
from surrogate import ModelWrapper
from predictor import Scalar
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

class MockModel(torch.nn.Module):
    """Mock model for testing ModelWrapper"""
    def __init__(self, A, L, dropout_rate=0.5):
        super().__init__()
        self.A = A
        self.L = L
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(A*L, 1)
        )
        # Initialize weights for predictable behavior
        torch.nn.init.ones_(self.layers[1].weight)
        torch.nn.init.zeros_(self.layers[1].bias)
        
    def forward(self, x):
        assert x.shape[1] == self.A, f'X shape wrong {x.shape[1]=} not {self.A=}'
        assert x.shape[2] == self.L, f'X shape wrong {x.shape[2]=} not {self.L=}'
        x = x.flatten(start_dim=1)
        assert x.shape[1] == self.L*self.A, f'X shape wrong {x.shape[1]=} not {self.L*self.A=}'

        # assert self.layers[1].data.weight.shape[0] == self.L*self.A, f'X shape wrong {self.layers[1].weights.shape[0]=} not {self.L*self.A=}'
        return self.layers(x)

class MockEnsemble(torch.nn.ModuleList):
    """Mock ensemble model for testing ModelWrapper"""
    def __init__(self, A, L, n_models=5, variance=0.1):
        super().__init__()
        self.A = A
        self.L = L
        
        self.models = torch.nn.ModuleList([
            MockModel(A, L) for i in range(n_models)
        ])
        # for model in self.models:
        #     # Initialize weights with variance
        #     model.weight.data = torch.randn_like(model.weight.data) * variance + 1.0
        #     model.bias.data.fill_(0.0)
            
    def forward(self, x):
        assert x.shape[1] == self.A, f'X shape wrong {x.shape[1]=} not {self.A=}'
        assert x.shape[2] == self.L, f'X shape wrong {x.shape[2]=} not {self.L=}'
        return torch.stack([model(x) for model in self.models]).mean(dim=0)

@pytest.mark.parametrize("uncertainty_method", [
    None,
    MCDropout(n_samples=10),
    DeepEnsemble()
])
def test_model_wrapper_predict(sample_data, uncertainty_method):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and wrapper
    if isinstance(uncertainty_method, DeepEnsemble):
        model = MockEnsemble(A=A, L=L).to(device)
    else:
        model = MockModel(A=A, L=L).to(device)
    
    wrapper = ModelWrapper(
        model=model,
        predictor=Scalar(),
        uncertainty_method=uncertainty_method
    )
    
    # Test predictions
    predictions = wrapper.predict(sample_data)
    
    # Test output shape
    assert predictions.shape == (N,1), "Prediction shape mismatch"
    assert predictions.dtype == torch.float32, "Prediction dtype mismatch"
    
    

@pytest.mark.parametrize("uncertainty_method", [
    MCDropout(n_samples=10),
    DeepEnsemble()
])
def test_model_wrapper_uncertainty(sample_data, uncertainty_method):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and wrapper
    if isinstance(uncertainty_method, DeepEnsemble):
        model = MockEnsemble(A=A, L=L).models.to(device)
    else:
        model = MockModel(A=A, L=L).to(device)
    
    wrapper = ModelWrapper(
        model=model,
        predictor=Scalar(),
        uncertainty_method=uncertainty_method
    )
    
    
    # Test uncertainty estimates
    uncertainties = wrapper.uncertainty(sample_data)
    
    # Test output shape
    assert uncertainties.shape == (N,), "Uncertainty shape mismatch"
    assert uncertainties.dtype == torch.float32, "Uncertainty dtype mismatch"
    
    # Test that uncertainties are non-negative
    assert torch.all(uncertainties >= 0), "Negative uncertainty values found"

# @pytest.mark.parametrize("batch_size", [1, 32, 64])
# def test_model_wrapper_batching(sample_data, batch_size):
#     N, A, L = sample_data.shape
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Test with both uncertainty methods
#     models_and_methods = [
#         (MockModel(A=A, L=L), MCDropout(n_samples=10)),
#         (MockEnsemble(A=A, L=L), DeepEnsemble())
#     ]
    
#     for model, uncertainty_method in models_and_methods:
#         model = model.to(device)
#         wrapper = ModelWrapper(
#             model=model,
#             predictor=Scalar(),
#             uncertainty_method=uncertainty_method
#         )
        
#         # Get full predictions and uncertainties
#         predictions = wrapper.predict(sample_data)
#         uncertainties = wrapper.uncertainty(sample_data)
        
#         # Get batched predictions and uncertainties
#         predictions_batched = []
#         uncertainties_batched = []
#         for i in range(0, N, batch_size):
#             batch = sample_data[i:i+batch_size]
#             predictions_batched.append(wrapper.predict(batch))
#             uncertainties_batched.append(wrapper.uncertainty(batch))
        
#         predictions_batched = torch.cat(predictions_batched)
#         uncertainties_batched = torch.cat(uncertainties_batched)
        
#         # Test batched results match full results (within numerical tolerance)
#         assert torch.allclose(predictions, predictions_batched, rtol=1e-4), "Batched predictions don't match"
#         assert torch.allclose(uncertainties, uncertainties_batched, rtol=1e-4), "Batched uncertainties don't match"

def test_model_wrapper_no_uncertainty_error(sample_data):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wrapper without uncertainty method
    model = MockModel(A=A, L=L).to(device)
    wrapper = ModelWrapper(
        model=model,
        predictor=Scalar(),
        uncertainty_method=None
    )
    
    # Test that calling uncertainty raises error
    with pytest.raises(ValueError):
        wrapper.uncertainty(sample_data) 