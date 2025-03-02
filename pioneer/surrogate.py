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

    def predict(self, x, batch_size=32):
        """Generate predictions using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
            
        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,) for single task
            or (N, T) for T tasks
        """
        predictions = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU
                batch_x = batch_x.to(self.device)
                
                if isinstance(self.model, list):
                    # Handle ensemble of models
                    batch_preds = torch.stack([
                        self.predictor.predict(m, batch_x)
                        for m in self.model
                    ])
                    # Average predictions across ensemble and move to CPU
                    pred = batch_preds.mean(dim=0).cpu()
                else:
                    # Single model prediction
                    pred = self.predictor.predict(self.model, batch_x).cpu()
                
                predictions.append(pred)
                
        return torch.cat(predictions, dim=0)

    def uncertainty(self, x, batch_size=32):
        """Generate uncertainty estimates using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
            
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
            
        # Initialize output tensor on CPU
        uncertainties = torch.empty(x.size(0))
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        start_idx = 0
        for batch_x, in loader:
            # Move batch to GPU, get uncertainty, move back to CPU
            batch_x = batch_x.to(self.device)
            uncertainty = self.uncertainty_method.estimate(self.model, batch_x).cpu()
            
            # Store in the pre-allocated tensor
            end_idx = start_idx + uncertainty.size(0)
            uncertainties[start_idx:end_idx] = uncertainty
            start_idx = end_idx
            
        return uncertainties




        