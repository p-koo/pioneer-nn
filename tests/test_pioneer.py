import torch
import pytest
import pytorch_lightning as pl
import sys
sys.path.append('./')
from pioneer import PIONEER
from pioneer.surrogate import ModelWrapper
from pioneer.oracle import SingleOracle
from pioneer.generator import RandomGenerator
from pioneer.acquisition import RandomAcquisition
from pioneer.predictor import Scalar

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
def test_pioneer_initialization(sample_data, batch_size, tmp_path):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=weight_path,  # Use the temporary weight file
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=batch_size)
    
    # Initialize PIONEER
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        batch_size=batch_size
    )
    
    assert pioneer.batch_size == batch_size, "Batch size not set correctly"
    assert pioneer.model == model_wrapper.model, "Model not set correctly"
    assert pioneer.surrogate == model_wrapper, "Surrogate not set correctly"
    assert pioneer.oracle == oracle, "Oracle not set correctly"
    assert pioneer.generator == generator, "Generator not set correctly"
    assert pioneer.acquisition == acquisition, "Acquisition not set correctly"

@pytest.mark.parametrize("n_cycles", [1, 2])
def test_pioneer_training_loop_with_trainer(sample_data, n_cycles, tmp_path):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create initial training data
    y = torch.randn(N, 1)  # Random labels
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=weight_path,  # Use the temporary weight file
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        batch_size=32
    )
    
    # Run training cycles
    x_train, y_train = sample_data, y
    for cycle in range(n_cycles):
        x_new, y_new = pioneer.run_cycle(x_train, y_train, trainer_factory=lambda: pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False))
        
        # Test shapes
        assert x_new.shape[0] == N//2, f"Cycle {cycle}: Wrong number of selected sequences"
        assert x_new.shape[1:] == (A, L), f"Cycle {cycle}: Wrong sequence dimensions"
        assert y_new.shape[0] == N//2, f"Cycle {cycle}: Wrong number of labels"
        
        # Update training data
        x_train = torch.cat([x_train, x_new])
        y_train = torch.cat([y_train, y_new])

@pytest.mark.parametrize("n_cycles", [1, 2])
def test_pioneer_training_loop_with_training_fnc(sample_data, n_cycles, tmp_path):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create initial training data
    y = torch.randn(N, 1)  # Random labels
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=weight_path,  # Use the temporary weight file
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        batch_size=32
    )
    
    # Run training cycles
    x_train, y_train = sample_data, y
    for cycle in range(n_cycles):
        x_new, y_new = pioneer.run_cycle(x_train, y_train, training_fnc_enclosure=lambda: pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False).fit)
        
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

    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create components
    model = MockLightningModel(A, L)
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=weight_path,
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
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
def test_pioneer_dataloader_creation(sample_data, val_split, tmp_path):
    N, A, L = sample_data.shape
    y = torch.randn(N, 1)  # Random labels
    
    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create PIONEER instance
    model = MockLightningModel(A, L)
    pioneer = PIONEER(
        model=ModelWrapper(model=model, predictor=Scalar()),
        oracle=SingleOracle(
            model_class=MockModel,
            model_kwargs={'A': A, 'L': L},
            weight_path=weight_path,  # Use the temporary weight file
            predictor=Scalar()
        ),
        generator=RandomGenerator(),
        acquisition=RandomAcquisition(target_size=N//2),
        batch_size=32
    )
    
    # Test dataloader creation
    train_loader = pioneer._get_dataloader(sample_data, y)
    assert train_loader is not None, "Training dataloader not created"
    assert train_loader.batch_size == 32, "Wrong batch size"
    
    # Test with None inputs
    with pytest.raises(AssertionError):
        pioneer._get_dataloader(None, None)

@pytest.mark.parametrize("cold_start", [True, False])
def test_pioneer_cold_start(sample_data, tmp_path, cold_start):
    N, A, L = sample_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and save model weights
    weight_path = tmp_path / "model_weights.pt"
    model = MockModel(A=A, L=L)
    torch.save(model.state_dict(), weight_path)
    
    # Create components
    model = MockLightningModel(A, L)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    model_wrapper = ModelWrapper(model=model, predictor=Scalar())
    oracle = SingleOracle(
        model_class=MockModel,
        model_kwargs={'A': A, 'L': L},
        weight_path=weight_path,
        predictor=Scalar()
    )
    generator = RandomGenerator()
    acquisition = RandomAcquisition(target_size=N//2)
    
    # Initialize PIONEER with cold_start option
    pioneer = PIONEER(
        model=model_wrapper,
        oracle=oracle,
        generator=generator,
        acquisition=acquisition,
        batch_size=32,
        cold_start=cold_start
    )
    
    # Run a training cycle
    weights = []
    def logging_trainer(model, train_loader, val_loader):
        weights.append({k: v.clone() for k, v in model.state_dict().items()})
        pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False).fit(model, train_loader, val_loader)

    y = torch.randn(N, 1)  # Random labels
    x_new, y_new = pioneer.run_cycle(
        sample_data, 
        y, 
        training_fnc_enclosure=lambda:logging_trainer
    )
    
    final_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Test weight reset behavior
    if not cold_start:
        assert not hasattr(pioneer, 'initial_state'), "warm start should not store initial weights"
        assert all([torch.allclose(initial_weights[k], weights[0][k]) 
                      for k in initial_weights.keys()]), "In the first cycle weights shouldnt be different yet"
        
        # Run another cycle to test weight reset
        x_new2, y_new2 = pioneer.run_cycle(
            torch.cat([sample_data, x_new]), 
            torch.cat([y, y_new]),
            training_fnc_enclosure=lambda:logging_trainer
        )
        
        
        
        # Test that weights were reset to initial values before second training
        assert not all(torch.allclose(initial_weights[k], weights[1][k]) 
                      for k in initial_weights.keys()), "In the second cycle inital weights should be different from original weights"
        
        # Test that weights were reset to initial values before second training
        assert all(torch.allclose(final_weights[k], weights[1][k]) 
                      for k in initial_weights.keys()), "In the second cycle final weights should be the same as the weights after the first cycle"
        
    else:
        assert hasattr(pioneer, 'initial_state'), "cold start should store initial weights"
        assert all(torch.allclose(initial_weights[k], weights[0][k]) 
                      for k in initial_weights.keys()), "In the first cycle weights shouldnt be different yet"
        
        # Run another cycle to test weight reset
        x_new2, y_new2 = pioneer.run_cycle(
            torch.cat([sample_data, x_new]), 
            torch.cat([y, y_new]),
            training_fnc_enclosure=lambda:logging_trainer
        )
        
        
        
        # Test that weights were reset to initial values before second training
        assert all(torch.allclose(initial_weights[k], weights[1][k]) 
                      for k in initial_weights.keys()), "In the second cycle inital weights should be the same as original weights"
        
        # Test that weights were reset to initial values before second training
        assert not all(torch.allclose(final_weights[k], weights[1][k]) 
                      for k in initial_weights.keys()), "In the second cycle final weights should be the different from the weights after the first cycle"
        