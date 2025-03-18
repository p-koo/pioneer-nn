import torch
import pytest
import sys
sys.path.append('./pioneer')
from acquisition import RandomAcquisition, ScoreAcquisition, PoolBasedAcquisition

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



@pytest.mark.parametrize("target_size", [1, 10, 15])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_random_acquisition(sample_data, target_size, seed):
    acq = RandomAcquisition(target_size=target_size, seed=seed)
    
    # Test shapes
    N, A, L = sample_data.shape
    if N < target_size:
        with pytest.raises(AssertionError):
            selected, indices = acq(sample_data)
    else:
        selected, indices = acq(sample_data)
    
        assert selected.shape == (target_size, A, L), "Selected shape mismatch"
        assert indices.shape == (target_size,), "Indices shape mismatch"
        assert selected.dtype == sample_data.dtype, "Selected dtype mismatch"
        assert indices.dtype == torch.long, "Indices dtype mismatch"

        # Indices should be unique and match the selected sequences
        assert torch.unique(indices).numel() == target_size, "Indices are not unique"
        assert torch.all(sample_data[indices] == selected), "Indices do not match selected sequences"
        
        # Test selection is random but consistent with seed
        selected2, indices2 = acq(sample_data)
        assert not torch.equal(indices, indices2), "Different random selections with a new selection"
        
        acq_seeded = RandomAcquisition(target_size=target_size, seed=seed)
        selected3, indices3 = acq_seeded(sample_data)
        assert torch.equal(indices, indices3), "Same random selections with the same seed"

@pytest.mark.parametrize("target_size", [1, 10, 15])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_score_acquisition(sample_data, target_size, seed):
    class MockModel(torch.nn.Module):
        """
        Mock model for testing score acquisition
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
        

    N, A, L = sample_data.shape

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mock_model = MockModel(A=A, L=L).to(device)
    acq = ScoreAcquisition(target_size=target_size, scorer=mock_model.forward, device=device)
    
    # Test shapes
    if sample_data.shape[0] < target_size:
        with pytest.raises(AssertionError):
            selected, indices = acq(sample_data)
    else:
        selected, indices = acq(sample_data)
    
        assert selected.shape == (target_size, A, L), "Selected shape mismatch"
        assert indices.shape == (target_size,), "Indices shape mismatch"
        assert selected.dtype == sample_data.dtype, "Selected dtype mismatch"
        assert indices.dtype == torch.long, "Indices dtype mismatch"

        # Indices should be unique and match the selected sequences
        assert torch.unique(indices).numel() == target_size, "Indices are not unique"
        assert torch.all(sample_data[indices] == selected), "Indices do not match selected sequences"
        
        # Test selection matches the highest scores
        scores = mock_model.forward(sample_data).squeeze()
        _, sorted_indices = torch.sort(scores, descending=True)
        assert torch.equal(indices, sorted_indices[:target_size]), "Indices do not match the highest scores"

@pytest.mark.parametrize("pool_size", [100, 200])
@pytest.mark.parametrize("target_size", [10, 20])
def test_pool_based_acquisition_random(sample_data, pool_size, target_size):
    """Test PoolBasedAcquisition with RandomAcquisition selector"""
    N, A, L = sample_data.shape
    
    # Create larger pool of sequences
    pool = torch.zeros(pool_size, A, L)
    for i in range(pool_size):
        nt_idx = torch.randint(0, A, (L,))
        pool[i, nt_idx, torch.arange(L)] = 1

    # Insert training sequences at known positions
    prior_idx = torch.randperm(pool_size)[:N]
    pool[prior_idx] = sample_data
    
    # Create random selector
    random_selector = RandomAcquisition(target_size=target_size)
    
    # Create pool-based acquisition
    pool_acquisition = PoolBasedAcquisition(
        selector=random_selector,
        pool=pool,
        prior_idx=prior_idx,
        check_prior_idx=True
    )
    
    # Select sequences
    selected_x, selected_idx = pool_acquisition(sample_data)
    
    # Test output shapes
    assert selected_x.shape == (target_size, A, L), "Wrong shape for selected sequences"
    assert selected_idx.shape == (target_size,), "Wrong shape for selected indices"
    
    # Test selected sequences are from pool
    idx_sorted = torch.argsort(selected_idx)
    assert torch.all(selected_x[idx_sorted] == pool[selected_idx[idx_sorted]]), "Selected sequences don't match pool sequences"

@pytest.mark.parametrize("pool_size", [100, 200])
def test_pool_based_acquisition_with_prior_idx(sample_data, pool_size):
    """Test PoolBasedAcquisition with prior indices"""
    N, A, L = sample_data.shape
    target_size = N // 2
    
    # Create pool that includes training sequences
    pool = torch.zeros(pool_size, A, L)
    for i in range(pool_size):
        nt_idx = torch.randint(0, A, (L,))
        pool[i, nt_idx, torch.arange(L)] = 1
    
    # Insert training sequences at known positions
    prior_idx = torch.randperm(pool_size)[:N]
    pool[prior_idx] = sample_data
    
    # Create random selector
    random_selector = RandomAcquisition(target_size=target_size)
    
    # Create pool-based acquisition
    pool_acquisition = PoolBasedAcquisition(
        selector=random_selector,
        pool=pool,
        prior_idx=prior_idx,
        check_prior_idx=True
    )
    
    # Select sequences
    selected_x, selected_idx = pool_acquisition(sample_data)
    
    # Test that selected sequences are not from training data
    for idx in selected_idx:
        assert idx not in prior_idx, "Selected sequence from training data"
    
    # Test that selected sequences are valid one-hot encodings
    assert torch.all(selected_x.sum(dim=1) == 1), "Selected sequences are not one-hot encoded"

    # Test that selected sequences are from pool
    idx_sorted = torch.argsort(selected_idx)
    assert torch.all(selected_x[idx_sorted] == pool[selected_idx[idx_sorted]]), "Selected sequences don't match pool sequences"

@pytest.mark.parametrize("scorer_type", ['uncertainty', 'prediction'])
def test_pool_based_acquisition_with_score_selector(sample_data, scorer_type):
    """Test PoolBasedAcquisition with ScoreAcquisition selector"""
    N, A, L = sample_data.shape
    pool_size = 100
    target_size = N // 2
    
    # Create pool
    pool = torch.zeros(pool_size, A, L)
    for i in range(pool_size):
        nt_idx = torch.randint(0, A, (L,))
        pool[i, nt_idx, torch.arange(L)] = 1

    cat_pool = torch.cat([sample_data, pool])
    assert cat_pool.shape[0] == N+pool_size, 'Cat failed'
    # pool_idx = torch.arange(len(pool))
    scramble_idx = torch.randperm(len(cat_pool))
    cat_pool = cat_pool[scramble_idx]
    prior_idx = torch.argsort(scramble_idx)[:len(sample_data)]

    assert torch.all(sample_data==cat_pool[prior_idx]), 'Prior idx doesnt recover sample data'
    # Create mock scorer
    def mock_scorer(x):
        if scorer_type == 'uncertainty':
            return torch.rand(len(x))  # Random uncertainty scores
        else:
            return torch.sum(x, dim=(1,2))  # Sum of one-hot values
    
    # Create score selector
    score_selector = ScoreAcquisition(
        target_size=target_size,
        scorer=mock_scorer,
        batch_size=32
    )
    
    # Create pool-based acquisition
    pool_acquisition = PoolBasedAcquisition(
        selector=score_selector,
        pool=cat_pool,
        prior_idx=prior_idx,
        check_prior_idx=True
    )
    
    # Select sequences
    selected_x, selected_idx = pool_acquisition(sample_data)
    
    # Test output shapes
    assert selected_x.shape == (target_size, A, L), "Wrong shape for selected sequences"
    assert selected_idx.shape == (target_size,), "Wrong shape for selected indices"
    
    # Test selected sequences are from pool
    idx_sorted = torch.argsort(selected_idx)
    assert torch.all(selected_x[idx_sorted] == cat_pool[selected_idx[idx_sorted]]), "Selected sequences don't match pool sequences"

def test_pool_based_acquisition_invalid_prior_idx(sample_data):
    """Test PoolBasedAcquisition with invalid prior indices"""
    N, A, L = sample_data.shape
    pool_size = 100
    target_size = N // 2
    
    # Create pool
    pool = torch.zeros(pool_size, A, L)
    for i in range(pool_size):
        nt_idx = torch.randint(0, A, (L,))
        pool[i, nt_idx, torch.arange(L)] = 1
    
    # Create invalid prior indices (pointing to wrong sequences)
    prior_idx = torch.randperm(pool_size)[:N]
    
    # Create random selector
    random_selector = RandomAcquisition(target_size=target_size)
    
    # Create pool-based acquisition
    pool_acquisition = PoolBasedAcquisition(
        selector=random_selector,
        pool=pool,
        prior_idx=prior_idx,
        check_prior_idx=True
    )
    
    # Test that using invalid prior indices raises error
    with pytest.raises(ValueError):
        selected_x, selected_idx = pool_acquisition(sample_data)
