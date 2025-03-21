import torch
from torch.utils.data import DataLoader, TensorDataset


class UncertaintyMethod:
    """Abstract base class for uncertainty estimation methods.
    
    All uncertainty methods should inherit from this class and implement
    the __call__ method to estimate prediction uncertainties.
    """
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates for input sequences.
        
        Parameters
        ----------
        model : torch.nn.Module or list[torch.nn.Module]
            PyTorch model or list of models for ensemble prediction
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,) for single task
            or (N, T) for T tasks
        """
        pass


class MCDropout(UncertaintyMethod):
    """Uncertainty estimation using Monte Carlo Dropout.
    
    This class implements uncertainty estimation by performing multiple forward passes
    through a model with dropout enabled. The variance across predictions provides
    an estimate of model uncertainty.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of forward passes with dropout enabled, by default 20
            
    Examples
    --------
    >>> uncertainty = MCDropout(n_samples=20)
    >>> scores = uncertainty(model, sequences)
    """
    def __init__(self, n_samples=20):
        self.n_samples = n_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using MC Dropout.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model with dropout layers
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,) for single task
            or (N, T) for T tasks
        """
        model.train()  # Enable dropout
        
        x = x.to(self.device)   
                
        # Multiple forward passes with dropout
        preds = torch.stack([
            model(x) for _ in range(self.n_samples)
        ])
                
        # Calculate uncertainty and move to CPU
        uncertainty = torch.std(preds, dim=0).squeeze(-1)
                
        return uncertainty

class DeepEnsemble(UncertaintyMethod):
    """Uncertainty estimation using Deep Ensembles.
    
    This class implements uncertainty estimation by combining predictions from multiple
    independently trained models. The variance across model predictions provides
    an estimate of model uncertainty.
    
    Examples
    --------
    >>> models = [model1, model2, model3]
    >>> uncertainty = DeepEnsemble()
    >>> scores = uncertainty(models, sequences)
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __call__(self, models: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using model ensemble.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model containing a list of models in its .models attribute
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,) for single task
            or (N, T) for T tasks
        """
        [model.eval() for model in models]

        x = x.to(self.device)
                
        # Get predictions from all models
        preds = torch.stack([
            model(x) for model in models
        ])
                
        # Calculate uncertainty and move to CPU
        uncertainty = torch.std(preds, dim=0).squeeze(-1)
                
        return uncertainty
