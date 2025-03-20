import torch
import pytest
import sys
sys.path.append('./pioneer')
from attribution import Saliency
from surrogate import ModelWrapper
from predictor import Scalar
from uncertainty import MCDropout, DeepEnsemble
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
            self.models = torch.nn.ModuleList([MockModel(A,L,fixed_weights=fixed_weights) for _ in range(10)])
            if fixed_weights:
                # Initialize weights for predictable behavior
                for model in self.models:
                    torch.nn.init.ones_(model.linear.weight)
                    torch.nn.init.zeros_(model.linear.bias)
        elif uncertainty_method == "dropout":
            self.linear = torch.nn.Linear(A*L, 1)
            if fixed_weights:
                torch.nn.init.ones_(self.linear.weight)
                torch.nn.init.zeros_(self.linear.bias)
            self.dropout = torch.nn.Dropout(p=0.5)

        else:
            self.linear = torch.nn.Linear(A*L, 1)
            if fixed_weights:
                torch.nn.init.ones_(self.linear.weight)
                torch.nn.init.zeros_(self.linear.bias)

        self.uncertainty_method = uncertainty_method
        
    def forward(self, x):
        if self.uncertainty_method == "ensemble":
            return torch.stack([model(x.flatten(start_dim=1)) for model in self.models]).mean(dim=0)
        elif self.uncertainty_method == "dropout":
            return self.dropout(self.linear(x.flatten(start_dim=1)))
        else:
            return self.linear(x.flatten(start_dim=1))
    
@pytest.mark.parametrize("fixed_weights", [False,True])
@pytest.mark.parametrize("unc_method", [None, "ensemble", "dropout"])
def test_saliency_with_model_wrapper(sample_data,fixed_weights,unc_method):
    """Test Saliency attribution with ModelWrapper"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and wrapper
    if unc_method == "ensemble":
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model.models, predictor=Scalar(),uncertainty_method=DeepEnsemble())
    elif unc_method == "dropout":
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar(),uncertainty_method=MCDropout())
    else:
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    # Create saliency attribution method
    if unc_method is None:
        saliency = Saliency(lambda x: model_wrapper.predict(x, auto_batch=False))
    else:
        saliency = Saliency(lambda x: model_wrapper.uncertainty(x, auto_batch=False))
    
    # Get attribution scores
    attr_scores = saliency(sample_data)
    
    # Test output shape
    assert attr_scores.shape == sample_data.shape, "Attribution scores shape mismatch"
    assert attr_scores.dtype == torch.float32, "Attribution scores dtype mismatch"
    
    # Test gradients are non-zero (since we initialized weights to ones)
    if not(unc_method == "ensemble" and fixed_weights):
        assert not torch.allclose(attr_scores, torch.zeros_like(attr_scores)), "Attribution scores are all zero"

@pytest.mark.parametrize("fixed_weights", [False,True])
@pytest.mark.parametrize("unc_method", [None, "ensemble", "dropout"])
def test_saliency_gradient_flow(sample_data,fixed_weights,unc_method):
    """Test that gradients flow correctly through the attribution process"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and wrapper
    if unc_method == "ensemble":
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model.models, predictor=Scalar(),uncertainty_method=DeepEnsemble())
    elif unc_method == "dropout":
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar(),uncertainty_method=MCDropout())
    else:
        model = MockModel(A=A, L=L,uncertainty_method=unc_method,fixed_weights=fixed_weights).to(device)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    # Create saliency attribution method
    if unc_method is None:
        saliency = Saliency(lambda x: model_wrapper.predict(x, auto_batch=False))
    else:
        saliency = Saliency(lambda x: model_wrapper.uncertainty(x, auto_batch=False))
    
    # Get attribution scores
    attr_scores = saliency(sample_data)
    
    # Test that gradients were computed
    assert attr_scores is not None, "No gradients computed for input"

@pytest.mark.parametrize("fixed_weights", [False,True])
def test_saliency_with_ensemble(sample_data,fixed_weights):
    """Test Saliency attribution with ensemble of models"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create ensemble of models
    n_models = 3
    models = [MockModel(A=A, L=L,fixed_weights=fixed_weights).to(device) for _ in range(n_models)]
    
    # Create wrapper with ensemble
    model_wrapper = ModelWrapper(model=models, predictor=Scalar(),uncertainty_method=DeepEnsemble())
    
    # Create saliency attribution method
    saliency = Saliency(lambda x: model_wrapper.predict(x, auto_batch=False))
    
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
        single_saliency = Saliency(lambda x: single_wrapper.predict(x, auto_batch=False))
        individual_scores.append(single_saliency(sample_data))
    
    # Average individual scores should be close to ensemble scores
    avg_individual = torch.stack(individual_scores).mean(dim=0)
    assert torch.allclose(attr_scores, avg_individual, rtol=1e-5), \
        "Ensemble attribution doesn't match average of individual attributions" 
    

@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("temp", [0, 1e-17, 1.0, 2.0])
@pytest.mark.parametrize("window", [None, (2,27), (5,45)])
@pytest.mark.parametrize("unc_method", [None, "ensemble", "dropout"])
@pytest.mark.parametrize("fixed_weights", [False,True])
def test_guided_mutagenesis(sample_data, mut_rate, temp, window,unc_method,fixed_weights):
        
    x = sample_data
    # Initialize generator
    N, A, L = x.shape
    if unc_method == "ensemble":
        model = MockModel(A, L, uncertainty_method=unc_method, fixed_weights=fixed_weights)
        model_wrapper = ModelWrapper(model=model.models, predictor=Scalar(),uncertainty_method=DeepEnsemble())
    elif unc_method == "dropout":
        model = MockModel(A, L, uncertainty_method=unc_method, fixed_weights=fixed_weights)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar(),uncertainty_method=MCDropout())
    else:
        model = MockModel(A, L, uncertainty_method=unc_method, fixed_weights=fixed_weights)
        model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    
    if unc_method is None:
        saliency = Saliency(lambda x: model_wrapper.predict(x, auto_batch=False))
    else:
        saliency = Saliency(lambda x: model_wrapper.uncertainty(x, auto_batch=False))
    
    if window is not None:
        gen = GuidedMutagenesisGenerator(saliency, mut_rate=mut_rate, temp=temp, seed=42, mut_window=window)
    else:
        gen = GuidedMutagenesisGenerator(saliency, mut_rate=mut_rate, temp=temp, seed=42)
    
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

    
    # # Test if mutation is guided by attribution
    # attribution = saliency(x[0].unsqueeze(0))[0]
    # empirical_mut_rate = torch.logical_and((x == 0), (out ==1)).float().mean(dim=0)

    # if temp > 1e-17 and mut_rate < 0.5:
    #     threshold = None
    # elif temp < 1 and mut_rate > 0.5:
    #     threshold = 0.5
    # elif temp > 1e-17 and mut_rate < 0.75 and window is not None and window[1] - window[0] < 30:
    #     threshold = None
    # else:
    #     threshold = 0.1
    #     # Calculate Spearman correlation between empirical mutation rate and attribution scores
    # if window is None:
    #     # Calculate Spearman correlation between empirical mutation rate and attribution scores
    #     emp_flat = empirical_mut_rate.flatten().cpu().numpy()
    #     attr_flat = attribution.flatten().cpu().numpy()
    # else:
    #     # Calculate Spearman correlation between empirical mutation rate and attribution scores
    #     emp_flat = empirical_mut_rate[:,window[0]:window[1]].flatten().cpu().numpy()
    #     attr_flat = attribution[:,window[0]:window[1]].flatten().cpu().numpy()

    # if threshold is not None:
    #     correlation, _ = pearsonr(emp_flat, attr_flat)
    #     assert correlation > threshold, "Weak pearson correlation between mutations and attribution scores"


    #     correlation, _ = spearmanr(emp_flat, attr_flat)
    #     assert correlation > threshold, "Weak spearman correlation between mutations and attribution scores"
