import torch
import pytest
import sys
sys.path.append('./pioneer')
from generator import Random, Mutagenesis, GuidedMutagenesis
from attribution import Saliency

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

def test_random_generator(sample_data):
    # Test initialization
    for prob in [[0.25, 0.25, 0.25, 0.25], [0.3, 0.2, 0.2, 0.3], [0.1, 0.1, 0.1, 0.7]]:
        gen = Random(prob=prob, seed=42)
        assert torch.allclose(gen.prob, torch.tensor(prob)), "Initialization failed"
        
        # Test output shape
        N, A, L = sample_data.shape  # batch size, alphabet size, sequence length
        out = gen.generate(sample_data)
        assert out.shape == (N, A, L), "Output shape mismatch"
        assert out.dtype == torch.float32, "Output dtype mismatch"

        # Test one-hot property
        assert (out.sum(dim=1) == 1).all(), "One-hot property mismatch"
        # Check that all elements are either 0 or 1
        assert torch.all(torch.logical_or(out == 0, out == 1)), "Elements are not 0 or 1"
        
        # Test probabilities
        large_sample = gen.generate(torch.zeros(1000, A, L))
        empirical_probs = large_sample.mean(dim=(0,2))
        assert torch.allclose(empirical_probs, torch.tensor(prob), atol=0.05), "Empirical probabilities don't match expected probabilities"
        

# def test_mutagenesis_generator(sample_data):
#     for mut_rate in [0.1, 0.25, 0.5, 0.75, 1.0]:
#         gen = Mutagenesis(mut_rate=mut_rate, seed=42)
        
#         # Test shapes
#         N, A, L = sample_data.shape
#         out = gen.generate(sample_data)
#         assert out.shape == (N, A, L), "Output shape mismatch"
#         assert out.dtype == torch.float32, "Output dtype mismatch"
        
#         # Test mutation rate
#         empirical_mut_rate = ((sample_data != out).float().sum(dim=(1,2)) == 2).float().mean()
#         assert empirical_mut_rate.item() == mut_rate, "Mutation rate mismatch"
        
#        # Test one-hot property
#         assert (out.sum(dim=1) == 1).all(), "One-hot property mismatch"
#         # Check that all elements are either 0 or 1
#         assert torch.all(torch.logical_or(out == 0, out == 1)), "Elements are not 0 or 1"
        
#         # Test mutation window
#         gen_window = Mutagenesis(mut_rate=mut_rate, mut_window=(10,20), seed=42)
#         out_window = gen_window.generate(sample_data)

#         # Test one-hot property
#         assert (out_window.sum(dim=1) == 1).all(), "One-hot property mismatch"
#         # Check that all elements are either 0 or 1
#         assert torch.all(torch.logical_or(out_window == 0, out_window == 1)), "Elements are not 0 or 1"

#         # Test no mutations in other regions
#         assert torch.equal(sample_data[:,:,:10], out_window[:,:,:10]), "Out of window mutations"
#         assert torch.equal(sample_data[:,:,20:], out_window[:,:,20:]), "Out of window mutations"

#         # Test proper mutation rate in window
#         empirical_mut_rate = ((sample_data[:,:,10:20] != out_window[:,:,10:20]).float().sum(dim=(1,2)) == 2).float().mean()
#         assert empirical_mut_rate.item() == mut_rate, "Mutation rate mismatch"

# def test_guided_mutagenesis(sample_data):
#     # Mock attribution method
#     class MockAttribution:
#         def attribute(self, x):
#             return torch.ones_like(x)
    
#     gen = GuidedMutagenesis(MockAttribution(), mut_rate=0.1, seed=42)
    
#     # Test shapes
#     N, A, L = sample_data.shape
#     out = gen.generate(sample_data)
#     assert out.shape == (N, A, L)
#     assert out.dtype == torch.float32
    
#     # Test one-hot property
#     assert torch.allclose(out.sum(dim=1), torch.ones(N, L))
    
#     # Test mutation rate
#     diff = (sample_data != out).float().mean()
#     assert abs(diff.item() - 0.1) < 0.02