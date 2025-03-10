import torch
import pytest
import sys
sys.path.append('./pioneer')
from predictor import Scalar, Profile

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

class MockScalarModel(torch.nn.Module):
    """Mock model that outputs scalar values"""
    def __init__(self, A, L, n_tasks=1):
        super().__init__()
        self.A = A
        self.L = L
        self.n_tasks = n_tasks
        self.linear = torch.nn.Linear(A*L, n_tasks)
        
    def forward(self, x):
        out = self.linear(x.flatten(start_dim=1))
        if self.n_tasks > 1:
            return out.view(-1, self.n_tasks)
        return out.view(-1)

class MockProfileModel(torch.nn.Module):
    """Mock model that outputs profile predictions"""
    def __init__(self, A, L, n_tasks=1):
        super().__init__()
        self.A = A
        self.L = L
        self.n_tasks = n_tasks
        self.linear = torch.nn.Linear(A*L, L*n_tasks)
        
    def forward(self, x):
        out = self.linear(x.flatten(start_dim=1))
        if self.n_tasks > 1:
            return out.view(-1, self.n_tasks, self.L)
        return out.view(-1, self.L)

@pytest.mark.parametrize("task_index", [None, 0, 1, 2])
@pytest.mark.parametrize("n_tasks", [1, 3])
def test_scalar_predictor(sample_data, task_index, n_tasks):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Skip invalid combinations
    if task_index is not None and task_index >= n_tasks:
        return
        
    # Initialize model and predictor
    model = MockScalarModel(A=A, L=L, n_tasks=n_tasks).to(device)
    predictor = Scalar(task_index=task_index)
    
    # Get predictions
    preds = predictor.predict(model, sample_data)
    
    # Test output shape
    if task_index is not None:
        assert preds.shape == (N,), "Prediction shape mismatch for single task"
    else:
        expected_shape = (N,) if n_tasks == 1 else (N, n_tasks)
        assert preds.shape == expected_shape, "Prediction shape mismatch"
    
    # Test output values match expected linear transformation
    expected = model(sample_data)
    if task_index is not None and n_tasks > 1:
        expected = expected[:, task_index]
    elif n_tasks == 1:
        expected = expected.view(-1)
    assert torch.allclose(preds, expected), "Predictions don't match expected values"

@pytest.mark.parametrize("task_index", [None, 0, 1, 2])
@pytest.mark.parametrize("n_tasks", [1, 3])
@pytest.mark.parametrize("reduction", [None, torch.mean, torch.sum, lambda x, dim: torch.max(x, dim=dim)[0]])
def test_profile_predictor(sample_data, task_index, n_tasks, reduction):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Skip invalid combinations
    if task_index is not None and task_index >= n_tasks:
        return
        
    # Initialize model and predictor
    model = MockProfileModel(A=A, L=L, n_tasks=n_tasks).to(device)
    predictor = Profile(reduction=reduction, task_index=task_index)
    
    # Get predictions
    preds = predictor.predict(model, sample_data)
    
    # Test that output is a a torch tensor
    assert isinstance(preds, torch.Tensor), f'Prediction is type {type(preds)} not torch.Tensor'
    
    # Test output shape
    if task_index is not None or n_tasks == 1:
        assert preds.shape == (N,), "Prediction shape mismatch for single task"
    else:
        assert preds.shape == (N, n_tasks), "Prediction shape mismatch for multiple tasks"
    
    # Test output values match expected transformation
    model_out = model(sample_data)
    if n_tasks > 1:
        model_out = model_out.view(N, n_tasks, L)
        if reduction is None and task_index is not None:
            expected = torch.mean(model_out[:, task_index, :], dim=-1)
        elif reduction is None and task_index is None:
            expected = torch.mean(model_out, dim=-1)
        elif task_index is not None:
            model_out = model_out[:, task_index]
            expected = reduction(model_out, dim=-1)
        else:
            expected = reduction(model_out, dim=-1)
    else:
        model_out = model_out.view(N, L)
        if reduction is None:
            expected = torch.mean(model_out, dim=-1)
        else:
            expected = reduction(model_out, dim=-1)
    
    assert torch.allclose(preds, expected), "Predictions don't match expected values"

# @pytest.mark.parametrize("batch_size", [1, 32, 64])
# def test_predictor_batching(sample_data, batch_size):
#     N, A, L = sample_data.shape
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Test both predictor types
#     model_scalar = MockScalarModel(A=A, L=L).to(device)
#     model_profile = MockProfileModel(A=A, L=L).to(device)
    
#     predictor_scalar = Scalar()
#     predictor_profile = Profile(reduction=torch.mean)
    
#     # Get full predictions
#     preds_scalar = predictor_scalar.predict(model_scalar, sample_data)
#     preds_profile = predictor_profile.predict(model_profile, sample_data)
    
#     # Get batched predictions
#     preds_scalar_batched = []
#     preds_profile_batched = []
#     for i in range(0, N, batch_size):
#         batch = sample_data[i:i+batch_size]
#         preds_scalar_batched.append(predictor_scalar.predict(model_scalar, batch))
#         preds_profile_batched.append(predictor_profile.predict(model_profile, batch))
    
#     preds_scalar_batched = torch.cat(preds_scalar_batched)
#     preds_profile_batched = torch.cat(preds_profile_batched)
    
#     # Test batched results match full results
#     assert torch.allclose(preds_scalar, preds_scalar_batched), "Scalar batched predictions don't match"
#     assert torch.allclose(preds_profile, preds_profile_batched), "Profile batched predictions don't match" 