import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable
from surrogate import ModelWrapper
from oracle import SingleOracle
from generator import Generator
from acquisition import Acquisition

class PIONEER:
    """Framework for active learning with sequence generation.
    
    Parameters
    ----------
    model : ModelWrapper
        ModelWrapper instance for predictions and uncertainty
    oracle : Oracle
        Oracle instance for ground truth labels
    generator : Generator
        Generator instance for sequence proposals
    acquisition : Acquisition
        Acquisition instance for sequence selection
    batch_size : int, optional
        Batch size for dataloaders, by default 32
        
    Examples
    --------
    >>> pioneer = PIONEER(
    ...     model=ModelWrapper(...),
    ...     oracle=SingleOracle(...),
    ...     generator=MutationGenerator(mut_rate=0.1),
    ...     acquisition=UncertaintySelector(n_select=1000),
    ... )
    """
    def __init__(self, model: ModelWrapper, oracle: SingleOracle, generator: Generator, acquisition: Acquisition, batch_size: int = 32, num_workers: int = 0, cold_start: bool = False):
        self.surrogate = model
        self.model = self.surrogate.model
        self.oracle = oracle
        self.generator = generator
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cold_start = cold_start
        if self.cold_start:
            # Store initial model state
            self.initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_model(self,  trainer:Optional[pl.Trainer]=None, train_fnc:Optional[Callable]=None, 
                    train_loader:Optional[torch.utils.data.DataLoader]=None,x:Optional[torch.Tensor]=None, 
                    y:Optional[torch.Tensor]=None, val_loader:Optional[torch.utils.data.DataLoader]=None, 
                    val_x:Optional[torch.Tensor]=None, val_y:Optional[torch.Tensor]=None, **train_kwargs):
        """Train surrogate model on data.
        
        Parameters
        ----------
        trainer : pl.Trainer, optional
            PyTorch Lightning Trainer, by default None
            If None train_fnc must be provided
        train_fnc : function
            Function to train the model, by default None
            If trainer is provided, train_fnc is ignored
            must take model, train_loader, and an optional val_loader as arguments
        train_loader : torch.utils.data.DataLoader, optional
            Training dataloader, by default None
        x : torch.Tensor, optional
            Training sequences of shape (N, A, L), by default None
            Used if train_loader is None
        y : torch.Tensor, optional
            Training labels of shape (N,), by default None
            Used if train_loader is None
        val_loader : torch.utils.data.DataLoader, optional
            Validation dataloader, by default None
        val_x : torch.Tensor, optional
            Validation sequences of shape (N, A, L), by default None
            Used if val_loader is None
        val_y : torch.Tensor, optional
            Validation labels of shape (N,), by default None
            Used if val_loader is None
        """
        if self.cold_start:
            # Reset model weights
            self.model.load_state_dict(self.initial_state)
        
        # Create fresh dataloaders
        assert train_loader is not None or (x is not None and y is not None), 'train_loader or x and y must not be None'
        if train_loader is None:
            train_loader = self._get_dataloader(x, y)
        if val_loader is None:
            val_loader = self._get_dataloader(val_x, val_y) if val_x is not None else None
        
        # Train from scratch
        if trainer is not None:
            trainer.fit(self.model, train_loader, val_loader)
        else:
            train_fnc(self.model, train_loader, val_loader, **train_kwargs)

    def generate_sequences(self, x:torch.Tensor) -> torch.Tensor:
        """Generate new sequence proposals.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
                
        Returns
        -------
        torch.Tensor
            Generated sequences of shape (M, A, L)
        """
        return self.generator(x)

    def select_sequences(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select promising sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
                
        Returns
        -------
        torch.Tensor
            Selected sequences of shape (M, A, L)
        """
        return self.acquisition(x)

    def get_oracle_labels(self, x:torch.Tensor) -> torch.Tensor:
        """Get ground truth labels from oracle.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
                
        Returns
        -------
        torch.Tensor
            Oracle labels of shape (N,)
        """
        return self.oracle.predict(x, self.batch_size)

    def run_cycle(self, x:torch.Tensor, y:torch.Tensor, val_x:Optional[torch.Tensor]=None, val_y:Optional[torch.Tensor]=None, trainer_factory:Optional[Callable]=None, training_fnc_enclosure:Optional[Callable]=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run complete active learning cycle.
        
        Parameters
        ----------
        x : torch.Tensor
            Training sequences of shape (N, A, L)
        y : torch.Tensor
            Training labels of shape (N,)
        val_x : torch.Tensor, optional
            Validation sequences, by default None
        val_y : torch.Tensor, optional
            Validation labels, by default None
        trainer_factory : function, optional
            Function to create a trainer, by default None
            If None, training_fnc_enclosure must be provided
        training_fnc_enclosure : function, optional
            Function that returns a function to train the model
            Ignored if trainer_factory is provided
            Returned function must take model, train_loader, and an optional val_loader as arguments
                
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (M, A, L)
            - Their labels of shape (M,)
        """
        if trainer_factory is not None:
            trainer = trainer_factory()
            training_fnc=None
        else:
            trainer = None
            assert training_fnc_enclosure is not None, 'training_fnc_enclosure must be provided if trainer_factory is None'
            training_fnc = training_fnc_enclosure()

        self.train_model(trainer=trainer, train_fnc=training_fnc, x=x, y=y, val_x=val_x, val_y=val_y)
        generated_x = self.generate_sequences(x)
        selected_x, selected_idx = self.select_sequences(generated_x)
        selected_y = self.get_oracle_labels(selected_x)
        return selected_x, selected_y

    def _get_dataloader(self, x:torch.Tensor, y:torch.Tensor) -> torch.utils.data.DataLoader:
        """Create DataLoader from tensors."""
        assert (x is not None) and (y is not None), 'x and y must not be None'
        assert x.shape[0] == y.shape[0], 'x and y must have the same number of samples'
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    

    @staticmethod
    def save_weights(model:ModelWrapper, path:str, cycle:Optional[int]=None):
        """Save model weights to file.
        
        Parameters
        ----------
        model : ModelWrapper
            ModelWrapper instance to save
        path : str
            Path to save weights
        cycle : int, optional
            Current cycle number to append to filename, by default None
        """
        path = str(path)
        if not path.endswith('.pt'):
            path = path + '.pt'
        if cycle is not None:
            path = path.replace('.pt', f'_cycle{cycle}.pt')
        torch.save(model.state_dict(), path)