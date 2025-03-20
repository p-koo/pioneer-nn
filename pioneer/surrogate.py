import torch
from torch.utils.data import DataLoader, TensorDataset

class ModelWrapper(torch.nn.Module):
    """Model wrapper for predictions and uncertainty estimation.
    
    This class wraps a PyTorch model (or ensemble of models) to provide a unified interface
    for making predictions and estimating uncertainties. It handles model loading, device 
    placement, and batched inference.
    
    Parameters
    ----------
    model : torch.nn.Module or list[torch.nn.Module]
        Single model or list of models for ensemble prediction
    predictor : Predictor
        Prediction method for generating outputs from model(s)
    uncertainty_method : UncertaintyMethod, optional
        Method for estimating prediction uncertainty, by default None
        
    Examples
    --------
    >>> model = ModelWrapper(
    ...     model=MyModel(),
    ...     predictor=ScalarPredictor(),
    ...     uncertainty_method=MCDropout()
    ... )
    >>> predictions = model.predict(sequences)
    >>> uncertainties = model.uncertainty(sequences)
    """
    def __init__(self, model, predictor, uncertainty_method=None):
        super().__init__()
        self.model = model
        self.predictor = predictor
        self.uncertainty_method = uncertainty_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions using batched inference.
        
        This method handles both single model and ensemble prediction. For ensembles,
        predictions are averaged across all models.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,) for single task
            or (N, T) for T tasks, where T is number of tasks
        """
        
       
        # Move batch to GPU
        x = x.to(self.device)
        
        if isinstance(self.model, list):
            # Handle ensemble of models
            pred = torch.stack([
                self.predictor(m, x)
                for m in self.model
            ])
            # Average predictions across ensemble and move to CPU
            pred = pred.mean(dim=0)
        else:
            # Single model prediction
            pred = self.predictor(self.model, x)
        
                
        return pred

    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Generate uncertainty estimates using batched inference.
        
        This method uses the specified uncertainty estimation method (e.g. MC Dropout,
        ensemble variance) to compute uncertainty scores for each input sequence.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,), where higher values indicate
            greater prediction uncertainty
            
        Raises
        ------
        ValueError
            If no uncertainty method was specified during initialization
        """
        if self.uncertainty_method is None:
            raise ValueError("No uncertainty method specified")
            
       
        # Move batch to GPU, get uncertainty, move back to CPU
        x = x.to(self.device)
        uncertainty = self.uncertainty_method(self.model, x)
            
        return uncertainty



