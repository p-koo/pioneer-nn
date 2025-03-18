import torch
import pytest
import sys
sys.path.append('./pioneer')
from attribution import Saliency
from surrogate import ModelWrapper
from predictor import Scalar
from generator import GuidedMutagenesisGenerator
from math import floor
from scipy.stats import spearmanr, pearsonr

@pytest.fixture(params=[
    (10, 4, 100),  # original size
    (10, 4, 50),   # shorter sequence
    (32, 4, 100),  # one batch
    (46, 4, 100),  # 1.5 batches
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
    """Mock model for testing attribution methods"""
    def __init__(self, A, L, uncertainty_method=None, fixed_weights=False):
        super().__init__()
        self.A = A
        self.L = L
        # Simple linear layer for predictable gradients
        

        if uncertainty_method == "ensemble":
            self.models = torch.nn.ModuleList([torch.nn.Linear(A*L, 1) for _ in range(10)])
            if fixed_weights:
                # Initialize weights for predictable behavior
                for model in self.models:
                    torch.nn.init.ones_(model.weight)
                    torch.nn.init.zeros_(model.bias)
        elif uncertainty_method == "dropout":
            self.linear = torch.nn.Linear(A*L, 1)
            torch.nn.init.ones_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)
            self.dropout = torch.nn.Dropout(p=0.5)

        self.uncertainty_method = uncertainty_method
        
    def forward(self, x):
        if self.uncertainty_method == "ensemble":
            return torch.stack([model(x.flatten(start_dim=1)) for model in self.models]).mean(dim=0)
        elif self.uncertainty_method == "dropout":
            return self.dropout(self.linear(x.flatten(start_dim=1)))
        else:
            return self.linear(x.flatten(start_dim=1))
    

def test_saliency_with_model_wrapper(sample_data):
    """Test Saliency attribution with ModelWrapper"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and wrapper
    model = MockModel(A=A, L=L).to(device)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    # Create saliency attribution method
    saliency = Saliency(model_wrapper.predict)
    
    # Get attribution scores
    attr_scores = saliency(sample_data)
    
    # Test output shape
    assert attr_scores.shape == sample_data.shape, "Attribution scores shape mismatch"
    assert attr_scores.dtype == torch.float32, "Attribution scores dtype mismatch"
    
    # Test gradients are non-zero (since we initialized weights to ones)
    assert not torch.allclose(attr_scores, torch.zeros_like(attr_scores)), "Attribution scores are all zero"
    
    # Test that gradients are higher for positions with 1s in input
    # (due to linear model with all-ones weights)
    input_mask = (sample_data == 1)
    assert (attr_scores[input_mask].mean() > attr_scores[~input_mask].mean()), \
        "Attribution scores should be higher for active positions"

def test_saliency_batch_consistency(sample_data):
    """Test that Saliency gives consistent results across different batch sizes"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and wrapper
    model = MockModel(A=A, L=L).to(device)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    # Create saliency attribution method
    saliency = Saliency(model_wrapper.predict)
    
    # Get full attribution scores
    attr_scores_full = saliency(sample_data)
    
    # Get batched attribution scores
    batch_size = max(1, N//2)
    attr_scores_batched = []
    for i in range(0, N, batch_size):
        batch = sample_data[i:i+batch_size]
        attr_scores_batched.append(saliency(batch))
    attr_scores_batched = torch.cat(attr_scores_batched)
    
    # Test that batched results match full results
    assert torch.allclose(attr_scores_full, attr_scores_batched, rtol=1e-5), \
        "Batched attribution scores don't match full attribution scores"

def test_saliency_gradient_flow(sample_data):
    """Test that gradients flow correctly through the attribution process"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and wrapper
    model = MockModel(A=A, L=L).to(device)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    # Create saliency attribution method
    saliency = Saliency(model_wrapper.predict)
    
    # Track gradients for input
    x = sample_data.clone().requires_grad_(True)
    
    # Get attribution scores
    attr_scores = saliency(x)
    
    # Test that gradients were computed
    assert x.grad is not None, "No gradients computed for input"
    assert torch.allclose(x.grad, attr_scores), "Gradients don't match attribution scores"
    
    # Test gradient flow through the whole model
    loss = attr_scores.sum()
    loss.backward()
    
    # Check that model parameters received gradients
    assert all(p.grad is not None for p in model.parameters()), \
        "Not all model parameters received gradients"

def test_saliency_with_ensemble(sample_data):
    """Test Saliency attribution with ensemble of models"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create ensemble of models
    n_models = 3
    models = [MockModel(A=A, L=L).to(device) for _ in range(n_models)]
    
    # Create wrapper with ensemble
    model_wrapper = ModelWrapper(model=models, predictor=Scalar())
    
    # Create saliency attribution method
    saliency = Saliency(model_wrapper.predict)
    
    # Get attribution scores
    attr_scores = saliency(sample_data)
    
    # Test output shape
    assert attr_scores.shape == sample_data.shape, "Attribution scores shape mismatch"
    
    # Test that ensemble produces non-zero gradients
    assert not torch.allclose(attr_scores, torch.zeros_like(attr_scores)), \
        "Ensemble attribution scores are all zero"
    
    # Test that each model in ensemble contributes to gradients
    individual_scores = []
    for model in models:
        single_wrapper = ModelWrapper(model=model, predictor=Scalar())
        single_saliency = Saliency(single_wrapper.predict)
        individual_scores.append(single_saliency(sample_data))
    
    # Average individual scores should be close to ensemble scores
    avg_individual = torch.stack(individual_scores).mean(dim=0)
    assert torch.allclose(attr_scores, avg_individual, rtol=1e-5), \
        "Ensemble attribution doesn't match average of individual attributions" 
    

@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("temp", [0, 1e-17, 1.0, 2.0])
@pytest.mark.parametrize("window", [None, (2,27), (5,45)])
def test_guided_mutagenesis(sample_data_big, mut_rate, temp, window):
        
    x = sample_data_big
    # Initialize generator
    N, A, L = x.shape
    model = MockModel(A, L)
    if window is not None:
        gen = GuidedMutagenesisGenerator(model.attribute, mut_rate=mut_rate, temp=temp, seed=42, mut_window=window)
    else:
        gen = GuidedMutagenesisGenerator(model.attribute, mut_rate=mut_rate, temp=temp, seed=42)
    
    # Test shapes
    N, A, L = x.shape
    out = gen(x)
    assert out.shape == (N, A, L)
    assert out.dtype == torch.float32
    
    # Test one-hot property
    assert (out.sum(dim=1) == 1).all(), "One-hot property mismatch"
    # Check that all elements are either 0 or 1
    assert torch.all(torch.logical_or(out == 0, out == 1)), "Elements are not 0 or 1"
    
    if window is None:
        # Test mutation rate
        empirical_mut_rate = ((x != out).int().sum(dim=1) == 2).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*L)/L).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.05), "Mutation rate mismatch"

    else:
        # Test no mutations in other regions
        assert torch.equal(x[:,:,:window[0]], out[:,:,:window[0]]), "Out of window mutations"
        assert torch.equal(x[:,:,window[1]:], out[:,:,window[1]:]), "Out of window mutations"

        # Test proper mutation rate in window
        empirical_mut_rate = ((x[:,:,window[0]:window[1]] != out[:,:,window[0]:window[1]]).int().sum(dim=1) == 2).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*(window[1]-window[0]))/(window[1]-window[0])).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.05), "Attribution mismatch in window"

    
    # Test if mutation is guided by attribution
    attribution = model.attribute(x[0].unsqueeze(0))[0]
    empirical_mut_rate = torch.logical_and((x == 0), (out ==1)).float().mean(dim=0)

    if temp > 1e-17 and mut_rate < 0.5:
        threshold = None
    elif temp < 1 and mut_rate > 0.5:
        threshold = 0.5
    elif temp > 1e-17 and mut_rate < 0.75 and window is not None and window[1] - window[0] < 30:
        threshold = None
    else:
        threshold = 0.1
        # Calculate Spearman correlation between empirical mutation rate and attribution scores
    if window is None:
        # Calculate Spearman correlation between empirical mutation rate and attribution scores
        emp_flat = empirical_mut_rate.flatten().cpu().numpy()
        attr_flat = attribution.flatten().cpu().numpy()
    else:
        # Calculate Spearman correlation between empirical mutation rate and attribution scores
        emp_flat = empirical_mut_rate[:,window[0]:window[1]].flatten().cpu().numpy()
        attr_flat = attribution[:,window[0]:window[1]].flatten().cpu().numpy()

    if threshold is not None:
        correlation, _ = pearsonr(emp_flat, attr_flat)
        assert correlation > threshold, "Weak pearson correlation between mutations and attribution scores"


        correlation, _ = spearmanr(emp_flat, attr_flat)
        assert correlation > threshold, "Weak spearman correlation between mutations and attribution scores"
