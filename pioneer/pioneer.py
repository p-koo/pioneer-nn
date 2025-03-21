import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable
from pioneer.surrogate import ModelWrapper
from pioneer.oracle import SingleOracle
from pioneer.generator import Generator
from pioneer.acquisition import Acquisition

class PIONEER:
    """Framework for active learning with sequence generation.
    
    This class implements an active learning cycle that:
    1. Trains a surrogate model on labeled data
    2. Generates candidate sequences using the generator
    3. Selects promising sequences using the acquisition function
    4. Gets ground truth labels from the oracle
    
    The PIONEER framework enables iterative improvement of sequence design by combining:
    - A surrogate model that learns to predict sequence properties
    - A generator that proposes new candidate sequences
    - An acquisition function that selects promising candidates
    - An oracle that provides ground truth labels
    
    Parameters
    ----------
    model : pioneer.surrogate.ModelWrapper
        ModelWrapper instance containing the surrogate model, predictor and uncertainty estimator
    oracle : pioneer.oracle.Oracle
        Oracle instance that provides ground truth labels for sequences
    generator : pioneer.generator.Generator
        Generator instance that proposes new candidate sequences
    acquisition : pioneer.acquisition.Acquisition
        Acquisition instance that selects promising sequences for labeling
    batch_size : int, optional
        Batch size for training and inference, by default 32
    num_workers : int, optional
        Number of workers for data loading, by default 0
    cold_start : bool, optional
        Whether to reset model weights before each training cycle, by default False
        
    Examples
    --------
    >>> # Initialize PIONEER with components
    >>> pioneer = PIONEER(
    ...     model=ModelWrapper(
    ...         model=MyModel(),
    ...         predictor=ScalarPredictor(),
    ...         uncertainty_method=MCDropout()
    ...     ),
    ...     oracle=SingleOracle(oracle_model),
    ...     generator=MutationGenerator(mut_rate=0.1),
    ...     acquisition=UncertaintyAcquisition(n_select=1000),
    ...     batch_size=32,
    ...     cold_start=True
    ... )
    >>> # Run active learning cycle
    >>> x_new, y_new = pioneer.run_cycle(x_train, y_train)
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
        
        Provides flexibility in training approach by supporting either:
        1. PyTorch Lightning Trainer
        2. Custom training function
        3. Direct tensor inputs or DataLoaders
        
        The training can be customized through:
        - Using a PyTorch Lightning Trainer for full training control
        - Providing a custom training function for custom logic
        - Passing data directly as tensors or as DataLoaders
        
        Parameters
        ----------
        trainer : pl.Trainer, optional
            PyTorch Lightning Trainer for model training
            If provided, takes precedence over train_fnc
        train_fnc : Callable, optional
            Custom training function that takes (model, train_loader, val_loader, \**kwargs)
            Used only if trainer is None
        train_loader : torch.utils.data.DataLoader, optional
            DataLoader containing training data
        x : torch.Tensor, optional
            Training sequences of shape (N, A, L)
            Used to create train_loader if not provided
        y : torch.Tensor, optional
            Training labels of shape (N,) or (N, T) for T tasks
            Used to create train_loader if not provided
        val_loader : torch.utils.data.DataLoader, optional
            DataLoader containing validation data
        val_x : torch.Tensor, optional
            Validation sequences of shape (N, A, L)
            Used to create val_loader if not provided
        val_y : torch.Tensor, optional
            Validation labels of shape (N,) or (N, T)
            Used to create val_loader if not provided
        \**train_kwargs
            Additional keyword arguments passed to train_fnc
            
        Raises
        ------
        AssertionError
            If neither train_loader nor (x,y) pair is provided
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
        """Generate new sequence proposals using the generator.
        
        Uses the configured generator to propose new candidate sequences based on the input sequences.
        The generator may implement various strategies like random mutations, crossover, or learned generation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size
            A is alphabet size (e.g. 4 for DNA)
            L is sequence length
                
        Returns
        -------
        torch.Tensor
            Generated sequences of shape (M, A, L)
            where M may differ from N depending on the generator's strategy
        """
        return self.generator(x)

    def select_sequences(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select promising sequences using the acquisition function.
        
        Uses the configured acquisition function to select sequences for labeling.
        The acquisition strategy may consider factors like:
        - Model uncertainty
        - Expected improvement
        - Upper confidence bounds
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size
            A is alphabet size (e.g. 4 for DNA)
            L is sequence length
                
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (M, A, L)
            - Selection indices of shape (M,)
            where M is determined by the acquisition function's strategy
        """
        return self.acquisition(x)

    def get_oracle_labels(self, x:torch.Tensor) -> torch.Tensor:
        """Get ground truth labels from oracle.
        
        Queries the oracle (e.g. experimental assay or trusted model) to obtain
        ground truth labels for the input sequences. The oracle provides the true
        target values that the surrogate model aims to predict.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size
            A is alphabet size (e.g. 4 for DNA)
            L is sequence length
                
        Returns
        -------
        torch.Tensor
            Oracle labels of shape (N,) for single task
            or (N, T) for T tasks
        """
        return self.oracle.predict(x, self.batch_size)

    def run_cycle(self, x:torch.Tensor, y:torch.Tensor, val_x:Optional[torch.Tensor]=None, val_y:Optional[torch.Tensor]=None, trainer_factory:Optional[Callable]=None, training_fnc_enclosure:Optional[Callable]=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run complete active learning cycle.
        
        Executes the full active learning loop:
        1. Trains surrogate model on current data
        2. Generates candidate sequences
        3. Selects promising candidates
        4. Gets oracle labels for selected sequences
        
        This is the main method that ties together all components to iteratively
        improve sequence design through active learning.
        
        Parameters
        ----------
        x : torch.Tensor
            Training sequences of shape (N, A, L) where:
            N is batch size
            A is alphabet size (e.g. 4 for DNA)
            L is sequence length
        y : torch.Tensor
            Training labels of shape (N,) for single task
            or (N, T) for T tasks
        val_x : torch.Tensor, optional
            Validation sequences of same shape as x
        val_y : torch.Tensor, optional
            Validation labels of same shape as y
        trainer_factory : Callable, optional
            Function that returns a new pl.Trainer instance
            Takes precedence over training_fnc_enclosure
        training_fnc_enclosure : Callable, optional
            Function that returns a training function
            Training function should take (model, train_loader, val_loader)
            Used only if trainer_factory is None
                
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (M, A, L)
            - Their oracle labels of shape (M,) or (M, T)
            
        Raises
        ------
        AssertionError
            If neither trainer_factory nor training_fnc_enclosure is provided
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
        """Create DataLoader from input tensors.
        
        Utility method to wrap input tensors in a DataLoader for batch processing.
        Handles data shuffling and multi-processing loading.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences
        y : torch.Tensor
            Input labels
            
        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader wrapping the input tensors with configured batch size
            and number of workers
            
        Raises
        ------
        AssertionError
            If x or y is None or if they have mismatched first dimensions
        """
        assert (x is not None) and (y is not None), 'x and y must not be None'
        assert x.shape[0] == y.shape[0], 'x and y must have the same number of samples'
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    

    @staticmethod
    def save_weights(model:ModelWrapper, path:str, cycle:Optional[int]=None):
        """Save model weights to file.
        
        Utility method to save model weights with optional cycle number in filename.
        Useful for tracking model evolution across active learning cycles.
        
        Parameters
        ----------
        model : pioneer.surrogate.ModelWrapper
            ModelWrapper instance containing the model to save
        path : str
            Path where weights will be saved
            If doesn't end in .pt, extension will be added
        cycle : int, optional
            Current cycle number to append to filename
            If provided, adds _cycleN before extension
            
        Examples
        --------
        >>> # Save weights without cycle number
        >>> PIONEER.save_weights(model, "model_weights.pt")
        >>> # Save weights with cycle number
        >>> PIONEER.save_weights(model, "model_weights.pt", cycle=5)
        # Saves to "model_weights_cycle5.pt"
        """
        path = str(path)
        if not path.endswith('.pt'):
            path = path + '.pt'
        if cycle is not None:
            path = path.replace('.pt', f'_cycle{cycle}.pt')
        torch.save(model.state_dict(), path)