import torch
import pytest
import sys
sys.path.append('./pioneer')
from generator import RandomGenerator, MutagenesisGenerator, GuidedMutagenesisGenerator
from scipy.stats import spearmanr, pearsonr
from math import floor

@pytest.fixture(params=[
    (10, 4, 100),  # original size
    (10, 4, 50),    # shorter sequence
    (10, 4, 200),  # longer sequence
    (32, 4, 100),  # one batch
    (46, 4, 100),  # 1.5 batches
    (64, 4, 100),  # two batches
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

@pytest.fixture(params=[
    (200, 4, 100),  # original size
    (200, 4, 50),    # shorter sequence
    (200, 4, 200),  # longer sequence
])
def sample_data_big(request):
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

@pytest.mark.parametrize("prob", [[0.25, 0.25, 0.25, 0.25], [0.3, 0.2, 0.2, 0.3], [0.1, 0.1, 0.1, 0.7]])
@pytest.mark.parametrize("window", [None, (4,45), (2,47)])
def test_random_generator(sample_data, prob, window):
    # Test initialization
    if window is not None:
        gen = RandomGenerator(prob=prob, seed=42, mut_window=window)
    else:
        gen = RandomGenerator(prob=prob, seed=42)
    assert torch.allclose(gen.prob, torch.tensor(prob)), "Initialization failed"
    
    # Test output shape
    N, A, L = sample_data.shape  # batch size, alphabet size, sequence length
    out = gen(sample_data)
    assert out.shape == (N, A, L), "Output shape mismatch"
    assert out.dtype == torch.float32, "Output dtype mismatch"

    # Test one-hot property
    assert (out.sum(dim=1) == 1).all(), "One-hot property mismatch"
    # Check that all elements are either 0 or 1
    assert torch.all(torch.logical_or(out == 0, out == 1)), "Elements are not 0 or 1"
    
    if window is not None:
        # Test no mutations in other regions
        assert torch.equal(sample_data[:,:,:window[0]], out[:,:,:window[0]]), "Out of window mutations"
        assert torch.equal(sample_data[:,:,window[1]:], out[:,:,window[1]:]), "Out of window mutations"

        # Test probabilities
        empirical_probs = out[:,:,window[0]:window[1]].mean(dim=(0,2))
        assert torch.allclose(empirical_probs, torch.tensor(prob), atol=0.06), "Empirical probabilities don't match expected probabilities"
    else:
        # Test probabilities
        large_sample = gen(torch.zeros(1000, A, L))
        empirical_probs = large_sample.mean(dim=(0,2))
        assert torch.allclose(empirical_probs, torch.tensor(prob), atol=0.05), "Empirical probabilities don't match expected probabilities"

@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("window", [None, (4,45), (2,47)])
def test_mutagenesis_generator(sample_data, mut_rate, window):
    if window is not None:
        gen = MutagenesisGenerator(mut_rate=mut_rate, seed=42, mut_window=window)
    else:
        gen = MutagenesisGenerator(mut_rate=mut_rate, seed=42)
        
    # Test shapes
    N, A, L = sample_data.shape
    out = gen(sample_data)
    assert out.shape == (N, A, L), "Output shape mismatch"
    assert out.dtype == torch.float32, "Output dtype mismatch"

    # Test one-hot property
    assert (out.sum(dim=1) == 1).all(), "One-hot property mismatch"
    # Check that all elements are either 0 or 1
    assert torch.all(torch.logical_or(out == 0, out == 1)), "Elements are not 0 or 1"
    
    if window is not None:
        # Test mutation rate
        empirical_mut_rate = ((sample_data[:,:,window[0]:window[1]] != out[:,:,window[0]:window[1]]).int().sum(dim=1) == 2).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*(window[1]-window[0]))/(window[1]-window[0])).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.05), "Mutation rate mismatch in window"
        
    else:
        # Test mutation rate
        empirical_mut_rate = ((sample_data != out).int().sum(dim=1) == 2).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*L)/L).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.05), "Mutation rate mismatch"
    

    



# @pytest.mark.parametrize("mut_rate", [0.1])
# @pytest.mark.parametrize("temp", [0])
# @pytest.mark.parametrize("window", [None])
# @pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
# @pytest.mark.parametrize("temp", [0, 1e-17, 1.0, 2.0])
# @pytest.mark.parametrize("window", [None, (10,20), (15,30)])

# @pytest.mark.parametrize("mut_rate", [1])
# @pytest.mark.parametrize("temp", [0])
# @pytest.mark.parametrize("window", [(15,30)])
@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("temp", [0, 1e-17, 1.0, 2.0])
@pytest.mark.parametrize("window", [None, (2,27), (5,45)])
def test_guided_mutagenesis(sample_data_big, mut_rate, temp, window):
    # Mock attribution method
    class MockModel(torch.nn.Module):
        """
        Mock model for testing guided mutagenesis
        It will produce a linear relationship between input and output
        and a constant attribution score for each nucleotide
        """
        def __init__(self,A, L):
            super().__init__()
            self.A = A
            self.L = L
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(A*L, 1),
            )
            
        def forward(self, x):
            return self.layers(x.flatten(start_dim=1))
        
        def attribute(self, x):
            x.requires_grad_(True)
            y = self.forward(x)
            y.sum().backward()
            return x.grad.clone()
        
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
