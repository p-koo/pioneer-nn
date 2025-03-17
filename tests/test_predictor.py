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
    preds = predictor(model, sample_data)
    
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
    preds = predictor(model, sample_data)
    
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

class DeterministicMockModel(torch.nn.Module):
    """Mock model with deterministic weights for testing predictors"""
    def __init__(self, A, L, n_tasks=1, output_type='scalar'):
        super().__init__()
        self.A = A
        self.L = L
        self.n_tasks = n_tasks
        self.output_type = output_type
        
        if output_type == 'scalar':
            self.linear = torch.nn.Linear(A*L, n_tasks)
        else:  # profile
            self.linear = torch.nn.Linear(A*L, n_tasks*L)
            
        # Initialize weights and bias deterministically
        torch.nn.init.ones_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        if self.output_type == 'scalar':
            return self.linear(x.flatten(start_dim=1))
        else:
            return self.linear(x.flatten(start_dim=1)).view(-1, self.n_tasks, self.L)

def test_scalar_predictor_deterministic(sample_data):
    """Test scalar predictor with deterministic weights"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test single task
    model = DeterministicMockModel(A=A, L=L, n_tasks=1, output_type='scalar').to(device)
    predictor = Scalar()
    preds = predictor(model, sample_data)
    
    # With all weights=1 and bias=0, output should equal sum of inputs
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(preds, expected), "Single task predictions don't match expected values"
    
    # Test multi-task with task selection
    model = DeterministicMockModel(A=A, L=L, n_tasks=3, output_type='scalar').to(device)
    predictor = Scalar(task_index=1)
    preds = predictor(model, sample_data)
    
    # Each task should get same prediction since weights are all 1
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(preds, expected), "Multi-task predictions don't match expected values"

def test_profile_predictor_deterministic(sample_data):
    """Test profile predictor with deterministic weights"""
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test single task profile
    model = DeterministicMockModel(A=A, L=L, n_tasks=1, output_type='profile').to(device)
    predictor = Profile()
    preds = predictor(model, sample_data)
    
    # With all weights=1 and bias=0, each position in profile should equal sum across alphabet
    expected = sample_data.sum(dim=(1,2))
    assert torch.allclose(preds, expected), "Single task profile predictions don't match"
    
    # Test multi-task profile with custom reduction
    model = DeterministicMockModel(A=A, L=L, n_tasks=3, output_type='profile').to(device)
    predictor = Profile(reduction=torch.sum, task_index=1)
    preds = predictor(model, sample_data)
    
    # Sum reduction of profile should equal L times sum across alphabet
    expected = sample_data.sum(dim=1).sum(dim=1) * L
    assert torch.allclose(preds, expected), "Multi-task profile predictions don't match"


