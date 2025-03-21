import torch
import pytest
import sys
sys.path.append('./')
from pioneer.attribution import Saliency

@pytest.fixture(params=[
    (10, 4, 100),  # original size
    (10, 4, 50),    # shorter sequence
    (10, 4, 200),  # longer sequence
    (32, 4, 100),  # one batch
    (46, 4, 100),  # 1.5 batches
    (64, 4, 100),  # two batches
    (1,4,1)         # single position
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
    """
    Mock model for testing saliency attribution
    It will produce a linear relationship between input and output
    """
    def __init__(self, A, L):
        super().__init__()
        self.A = A
        self.L = L
        self.linear = torch.nn.Linear(A*L, 1)
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
    
class MockModel2(torch.nn.Module):
    """
    Mock model for testing saliency attribution
    It will produce a linear relationship between input and output
    with an uncertainty score calculated by deep ensemble
    """
    def __init__(self, A, L):
        super().__init__()
        self.A = A
        self.L = L
        self.models = torch.nn.ModuleList([MockModel(A, L) for _ in range(10)])
    
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=0).mean(dim=0)
    
    def uncertainty(self, x):
        return torch.stack([model(x) for model in self.models], dim=0).std(dim=0)
    

@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_saliency(sample_data, batch_size):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and saliency
    model = MockModel(A=A, L=L).to(device)
    saliency = Saliency(model.forward)
    
    # Get attribution scores
    attr_scores = saliency.attribute(sample_data)
    
    # Test output shape matches input
    assert attr_scores.shape == sample_data.shape, "Attribution shape mismatch"
    assert attr_scores.dtype == sample_data.dtype, "Attribution dtype mismatch"
    
    # Test that attribution scores match the model weights
    assert torch.allclose(attr_scores, model.linear.weight.reshape(A,L)[None,:,:].expand(N,A,L), atol=1e-6), "Attribution scores do not match model weights"
    
    
@pytest.mark.parametrize("score_method", ["Y", "uncertainty"])
def test_saliency_ensemble(sample_data, score_method):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model and saliency
    model = MockModel2(A=A, L=L).to(device)
    if score_method == "Y":
        saliency = Saliency(model.forward)

        # Get attribution scores
        attr_scores = saliency.attribute(sample_data)
        
        # Test that attribution scores match the std of the model weights
        weight_mean = torch.stack([model.linear.weight.reshape(A,L) for model in model.models], dim=0).mean(dim=0)
            
        assert torch.allclose(attr_scores, weight_mean[None,:,:].expand(N,-1,-1), atol=1e-6), "Attribution scores do not match model weight mean"

    elif score_method == "uncertainty":
        saliency = Saliency(model.uncertainty)
    
        # Get attribution scores
        attr_scores = saliency.attribute(sample_data)
        
        
    # Test output shape matches input
    assert attr_scores.shape == sample_data.shape, "Attribution shape mismatch"
    assert attr_scores.dtype == sample_data.dtype, "Attribution dtype mismatch"