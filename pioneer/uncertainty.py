import torch
from torch.utils.data import DataLoader, TensorDataset


class UncertaintyMethod:
    """Abstract base class for uncertainty estimation methods.
    
    All uncertainty methods should inherit from this class and implement
    the estimate method.
    """
    def estimate(self, model, x, batch_size=32):
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
        
    def estimate(self, model, x, batch_size=32):
        """Generate uncertainty estimates using MC Dropout.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model with dropout layers
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
        model.train()  # Enable dropout
        uncertainties = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU
                batch_x = batch_x.to(self.device)
                
                # Multiple forward passes with dropout
                preds = torch.stack([
                    model(batch_x) for _ in range(self.n_samples)
                ])
                
                # Calculate uncertainty and move to CPU
                uncertainty = torch.std(preds, dim=0).cpu()
                uncertainties.append(uncertainty)
                
        return torch.cat(uncertainties, dim=0)


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
        
    def estimate(self, models, x, batch_size=32):
        """Generate uncertainty estimates using model ensemble.
        
        Parameters
        ----------
        models : list[torch.nn.Module]
            List of PyTorch models
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
        [model.eval() for model in models]
        uncertainties = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU
                batch_x = batch_x.to(self.device)
                
                # Get predictions from all models
                preds = torch.stack([
                    model(batch_x) for model in models
                ])
                
                # Calculate uncertainty and move to CPU
                uncertainty = torch.std(preds, dim=0).cpu()
                uncertainties.append(uncertainty)
                
        return torch.cat(uncertainties, dim=0)
    

