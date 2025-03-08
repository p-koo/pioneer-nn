import torch
import pytest
import sys
sys.path.append('./pioneer')
from proposer import MultiProposer, SequentialProposer
from generator import RandomGenerator, MutagenesisGenerator
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

@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("window", [None, (2,27), (5,45)])
def test_MultiGenerator(sample_data, mut_rate, window):
    x = sample_data
    # Test MultiGenerator
    probs = [0.25, 0.25, 0.25, 0.25]
    if window is not None:
        multi_gen = MultiProposer([RandomGenerator(prob=probs, seed=42, mut_window=window), 
                                    MutagenesisGenerator(mut_rate=mut_rate, mut_window=window, seed=42)])
    else:
        multi_gen = MultiProposer([RandomGenerator(prob=probs, seed=42), 
                                    MutagenesisGenerator(mut_rate=mut_rate, seed=42)])
    
    out_multi = multi_gen(x)

    # Check shape matches
    N, A, L = x.shape
    assert out_multi.shape == (2*N, A, L), "Shape mismatch in MultiGenerator output"
    
    # Check one-hot property
    assert (out_multi.sum(dim=1) == 1).all(), "One-hot property mismatch in MultiGenerator"
    assert torch.all(torch.logical_or(out_multi == 0, out_multi == 1)), "Elements are not 0 or 1 in MultiGenerator"

    # check that the first half of the output is random and the second half is mutated
    if window is None:
        empirical_mut_rate = out_multi[:N].mean(dim=(0,2))
        assert torch.allclose(empirical_mut_rate, torch.tensor(probs), atol=0.05), "MultiGenerator random mutation rate mismatch"
        
        empirical_mut_rate= torch.logical_and((out_multi[N:]==1), (x==0)).any(dim=1).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*L)/L).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.01), "MultiGenerator mutation rate mismatch"
    else:
        # Test no mutations outside window
        assert torch.equal(x[:,:,:window[0]], out_multi[:N,:,:window[0]]) and torch.equal(x[:,:,:window[0]], out_multi[N:,:,:window[0]]), "Out of window mutations in MultiGenerator"
        assert torch.equal(x[:,:,window[1]:],out_multi[:N,:,window[1]:]) and torch.equal(x[:,:,window[1]:],out_multi[N:,:,window[1]:]), "Out of window mutations in MultiGenerator"
        
        # Test mutation rate in window
        empirical_mut_rate = out_multi[:N,:,window[0]:window[1]].mean(dim=(0,2))
        assert torch.allclose(empirical_mut_rate, torch.tensor(probs).float(), atol=0.05), "MultiGenerator random mutation rate mismatch in window"
        empirical_mut_rate = torch.logical_and((out_multi[N:,:,window[0]:window[1]]==1), (x[:,:,window[0]:window[1]]==0)).any(dim=1).float().mean()
        expected_mut_rate = torch.tensor(floor(mut_rate*(window[1]-window[0]))/int(window[1]-window[0])).float()
        assert torch.allclose(empirical_mut_rate, expected_mut_rate, atol=0.01), "MultiGenerator mutation rate mismatch in window"

@pytest.mark.parametrize("mut_rate", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("window", [None, (2,27), (5,45)])
def test_SequentialGenerator(sample_data, mut_rate, window):
    # Test SequentialGenerator
    x = sample_data
    if window is not None:
        seq_gen = SequentialProposer([MutagenesisGenerator(mut_rate=mut_rate, mut_window=window, seed=42), 
                                MutagenesisGenerator(mut_rate=mut_rate, mut_window=window, seed=42)]) 
    else:
        seq_gen = SequentialProposer([MutagenesisGenerator(mut_rate=mut_rate, seed=42), 
                                MutagenesisGenerator(mut_rate=mut_rate, seed=42)]) 
    out_seq = seq_gen(x)

    # Check shape matches
    assert out_seq.shape == x.shape, "Shape mismatch in SequentialGenerator output"
    
    # Check one-hot property
    assert (out_seq.sum(dim=1) == 1).all(), "One-hot property mismatch in SequentialGenerator"
    assert torch.all(torch.logical_or(out_seq == 0, out_seq == 1)), "Elements are not 0 or 1 in SequentialGenerator"

    if window is not None:
        # Test no mutations outside window
        assert torch.equal(x[:,:,:window[0]], out_seq[:,:,:window[0]]), "Out of window mutations in SequentialGenerator" 
        assert torch.equal(x[:,:,window[1]:], out_seq[:,:,window[1]:]), "Out of window mutations in SequentialGenerator"

    




