import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

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
    trainer : pl.Trainer
        PyTorch Lightning Trainer for model training
    batch_size : int, optional
        Batch size for dataloaders, by default 32
        
    Examples
    --------
    >>> trainer = pl.Trainer(max_epochs=100)
    >>> pioneer = PIONEER(
    ...     model=ModelWrapper(...),
    ...     oracle=SingleOracle(...),
    ...     generator=MutationGenerator(mut_rate=0.1),
    ...     acquisition=UncertaintySelector(n_select=1000),
    ...     trainer=trainer
    ... )
    """
    def __init__(self, model, oracle, generator, acquisition, trainer, batch_size=32):
        self.model = model
        self.oracle = oracle
        self.generator = generator
        self.acquisition = acquisition
        self.trainer = trainer
        self.batch_size = batch_size
        # Store initial model state
        self.initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    def train_model(self, x, y, val_x=None, val_y=None):
        """Train surrogate model on data.
        
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
        """
        # Reset model weights
        self.model.load_state_dict(self.initial_state)
        
        # Reset trainer state
        self.trainer.fit_loop.epoch_progress.reset()
        self.trainer.fit_loop.epoch_loop.reset()
        self.trainer.fit_loop.reset()
        
        # Create fresh dataloaders
        train_loader = self._get_dataloader(x, y)
        val_loader = self._get_dataloader(val_x, val_y) if val_x is not None else None
        
        # Train from scratch
        self.trainer.fit(self.model, train_loader, val_loader)

    def generate_sequences(self, x):
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
        return self.generator.generate(x)

    def select_sequences(self, x):
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
        return self.acquisition.select(x)

    def get_oracle_labels(self, x):
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

    def run_cycle(self, x, y, val_x=None, val_y=None):
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
                
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (M, A, L)
            - Their labels of shape (M,)
        """
        self.train_model(x, y, val_x, val_y)
        generated_x = self.generate_sequences(x)
        selected_x = self.select_sequences(generated_x)
        selected_y = self.get_oracle_labels(selected_x)
        return selected_x, selected_y

    def _get_dataloader(self, x, y):
        """Create DataLoader from tensors."""
        if x is None or y is None:
            return None
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    

    @staticmethod
    def save_weights(model, path, cycle=None):
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
        if cycle is not None:
            path = path.replace('.pt', f'_cycle{cycle}.pt')
        torch.save(model.state_dict(), path)