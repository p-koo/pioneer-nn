import torch
import pytest
import pytorch_lightning as pl
import sys
sys.path.append('./pioneer')
from pioneer import PIONEER
from surrogate import ModelWrapper
from oracle import SingleOracle
from generator import Random as RandomGenerator
from acquisition import Random as RandomAcquisition
from predictor import Scalar

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
    """Mock model for testing PIONEER"""
    def __init__(self, A, L):
        super().__init__()
        self.A = A
        self.L = L
        self.linear = torch.nn.Linear(A*L, 1)
        # Initialize weights for predictable behavior
        torch.nn.init.ones_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))

class MockLightningModel(pl.LightningModule):
    """Mock Lightning model for testing PIONEER"""
    def __init__(self, A, L):
        super().__init__()
        self.model = MockModel(A, L)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("val_split", [0.0, 0.2])
def test_pioneer_initialization(sample_data, batch_size, val_split):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=None,  # Mock oracle doesn't need weights
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=batch_size)
    trainer = pl.Trainer(max_epochs=1)
    
    # Initialize PIONEER
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        trainer=trainer,
        batch_size=batch_size
    )
    
    assert pioneer.batch_size == batch_size, "Batch size not set correctly"
    assert pioneer.model == model_wrapper, "Model wrapper not set correctly"
    assert pioneer.oracle == oracle, "Oracle not set correctly"
    assert pioneer.generator == generator, "Generator not set correctly"
    assert pioneer.acquisition == acquisition, "Acquisition not set correctly"
    assert pioneer.trainer == trainer, "Trainer not set correctly"

@pytest.mark.parametrize("n_cycles", [1, 2])
def test_pioneer_training_loop(sample_data, n_cycles):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create initial training data
    y = torch.randn(N, 1)  # Random labels
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=None,
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    trainer = pl.Trainer(max_epochs=1)
    
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        trainer=trainer,
        batch_size=32
    )
    
    # Run training cycles
    x_train, y_train = sample_data, y
    for cycle in range(n_cycles):
        x_new, y_new = pioneer.run_cycle(x_train, y_train)
        
        # Test shapes
        assert x_new.shape[0] == N//2, f"Cycle {cycle}: Wrong number of selected sequences"
        assert x_new.shape[1:] == (A, L), f"Cycle {cycle}: Wrong sequence dimensions"
        assert y_new.shape[0] == N//2, f"Cycle {cycle}: Wrong number of labels"
        
        # Update training data
        x_train = torch.cat([x_train, x_new])
        y_train = torch.cat([y_train, y_new])

def test_pioneer_save_weights(sample_data, tmp_path):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=None,
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    trainer = pl.Trainer(max_epochs=1)
    
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        trainer=trainer,
        batch_size=32
    )
    
    # Test saving weights
    weight_path = tmp_path / "model.pt"
    pioneer.save_weights(model_wrapper, weight_path)
    assert weight_path.exists(), "Weight file not created"
    
    # Test saving with cycle number
    pioneer.save_weights(model_wrapper, weight_path, cycle=1)
    cycle_path = tmp_path / "model_cycle1.pt"
    assert cycle_path.exists(), "Weight file with cycle number not created"

@pytest.mark.parametrize("val_split", [0.0, 0.2])
def test_pioneer_dataloader_creation(sample_data, val_split):
    N, A, L = sample_data.shape
    y = torch.randn(N, 1)  # Random labels
    
    # Create PIONEER instance
    model = MockLightningModel(A, L)
    pioneer = PIONEER(
        model=ModelWrapper(model=model, predictor=Scalar()),
        oracle=SingleOracle(
            model_class=MockModel,
            model_kwargs={'A': A, 'L': L},
            weight_path=None,
            predictor=Scalar()
        ),
        generator=RandomGenerator(),
        acquisition=RandomAcquisition(target_size=N//2),
        trainer=pl.Trainer(max_epochs=1),
        batch_size=32
    )
    
    # Test dataloader creation
    train_loader = pioneer._get_dataloader(sample_data, y)
    assert train_loader is not None, "Training dataloader not created"
    assert train_loader.batch_size == 32, "Wrong batch size"
    
    # Test with None inputs
    none_loader = pioneer._get_dataloader(None, None)
    assert none_loader is None, "None inputs should return None" 