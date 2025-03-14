import torch
from torch.utils.data import DataLoader, TensorDataset

class ModelWrapper:
    """Model wrapper for predictions and uncertainty estimation.
    
    Parameters
    ----------
    model : torch.nn.Module or list[torch.nn.Module]
        Model or list of models for ensemble
    predictor : Predictor
        Prediction method for generating outputs
    uncertainty_method : UncertaintyMethod, optional
        Method for estimating prediction uncertainty, by default None
        
    Examples
    --------
    >>> model = ModelWrapper(
    ...     model=MyModel(),
    ...     predictor=ScalarPredictor(),
    ...     uncertainty_method=MCDropout()
    ... )
    """
    def __init__(self, model, predictor, uncertainty_method=None):
        self.model = model
        self.predictor = predictor
        self.uncertainty_method = uncertainty_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,) for single task
            or (N, T) for T tasks
        """
        
       
        # Move batch to GPU
        x = x.to(self.device)
        
        if isinstance(self.model, list):
            # Handle ensemble of models
            pred = torch.stack([
                self.predictor.predict(m, x)
                for m in self.model
            ])
            # Average predictions across ensemble and move to CPU
            pred = pred.mean(dim=0)
        else:
            # Single model prediction
            pred = self.predictor.predict(self.model, x)
        
                
        return pred

    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,)
            
        Raises
        ------
        ValueError
            If no uncertainty method was specified during initialization
        """
        if self.uncertainty_method is None:
            raise ValueError("No uncertainty method specified")
            
       
        # Move batch to GPU, get uncertainty, move back to CPU
        x = x.to(self.device)
        uncertainty = self.uncertainty_method.estimate(self.model, x)
            
        return uncertainty




        