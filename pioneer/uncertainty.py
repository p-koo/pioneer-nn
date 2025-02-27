import torch
from torch.utils.data import DataLoader, TensorDataset


class UncertaintyMethod:
    """Abstract base class for uncertainty estimation methods.
    
    All uncertainty methods should inherit from this class and implement
    the estimate method.
    """
    def estimate(self, model, x, batch_size=32):
        """Generate uncertainty estimates for input sequences.
        
        Args:
            model: PyTorch model or list of models
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing
            
        Returns:
            torch.Tensor: Uncertainty scores for each sequence
        """
        pass


class MCDropout(UncertaintyMethod):
    """Uncertainty estimation using Monte Carlo Dropout.
    
    Args:
        n_samples (int, optional): Number of forward passes with dropout.
            Defaults to 20.
            
    Example:
        >>> uncertainty = MCDropout(n_samples=20)
        >>> scores = uncertainty.estimate(model, sequences)
    """
    def __init__(self, n_samples=20):
        self.n_samples = n_samples
        
    def estimate(self, model, x, batch_size=32):
        """Generate uncertainty estimates using MC Dropout.
        
        Args:
            model: PyTorch model with dropout layers
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Uncertainty scores of shape (N,)
        """
        model.train()  # Enable dropout
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        uncertainties = []
        
        with torch.no_grad():
            for batch in loader:
                # Multiple forward passes with dropout
                preds = torch.stack([
                    model(batch[0]) for _ in range(self.n_samples)
                ])
                uncertainties.append(torch.std(preds, dim=0))
                
        return torch.cat(uncertainties, dim=0)


class DeepEnsemble(UncertaintyMethod):
    """Uncertainty estimation using Deep Ensembles.
    
    Example:
        >>> models = [model1, model2, model3]
        >>> uncertainty = DeepEnsemble()
        >>> scores = uncertainty.estimate(models, sequences)
    """
    def estimate(self, models, x, batch_size=32):
        """Generate uncertainty estimates using model ensemble.
        
        Args:
            models (list): List of PyTorch models
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Uncertainty scores of shape (N,)
        """
        [model.eval() for model in models]
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        uncertainties = []
        
        with torch.no_grad():
            for batch in loader:
                # Get predictions from all models
                preds = torch.stack([
                    model(batch[0]) for model in models
                ])
                uncertainties.append(torch.std(preds, dim=0))
                
        return torch.cat(uncertainties, dim=0)
    

