import torch
from torch.utils.data import DataLoader, TensorDataset


class UncertaintyMethod:
    """Abstract base class for uncertainty estimation methods.
    
    All uncertainty methods should inherit from this class and implement
    the estimate method.
    """
    def estimate(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates for input sequences.
        
        Parameters
        ----------
        model : torch.nn.Module or list[torch.nn.Module]
            PyTorch model or list of models
        x : torch.Tensor
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,)
        """
        pass

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.estimate(x)


class MCDropout(UncertaintyMethod):
    """Uncertainty estimation using Monte Carlo Dropout.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of forward passes with dropout, by default 20
            
    Examples
    --------
    >>> uncertainty = MCDropout(n_samples=20)
    >>> scores = uncertainty.estimate(model, sequences)
    """
    def __init__(self, n_samples=20):
        self.n_samples = n_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def estimate(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using MC Dropout.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model with dropout layers
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,)
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

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.estimate(x)

class DeepEnsemble(UncertaintyMethod):
    """Uncertainty estimation using Deep Ensembles.
    
    Examples
    --------
    >>> models = [model1, model2, model3]
    >>> uncertainty = DeepEnsemble()
    >>> scores = uncertainty.estimate(models, sequences)
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def estimate(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using model ensemble.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model containing a list of models in its .models attribute
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,)
        """
        [model.eval() for model in model.models]

        x = x.to(self.device)
                
        # Get predictions from all models
        preds = torch.stack([
            model(x) for model in model.models
        ])
                
        # Calculate uncertainty and move to CPU
        uncertainty = torch.std(preds, dim=0).squeeze(-1)
                
        return uncertainty
    
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.estimate(x)
