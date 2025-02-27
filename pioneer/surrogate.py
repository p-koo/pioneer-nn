import torch

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

    def predict(self, x, batch_size=32):
        """Generate predictions for input sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for processing, by default 32
            
        Returns
        -------
        torch.Tensor
            Model predictions
        """
        if isinstance(self.model, list):
            preds = [self.predictor.predict(m, x, batch_size) for m in self.model]
            return torch.stack(preds).mean(dim=0)
        return self.predictor.predict(self.model, x, batch_size)

    def uncertainty(self, x, batch_size=32):
        """Generate uncertainty estimates.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for processing, by default 32
            
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
        return self.uncertainty_method.estimate(self.model, x, batch_size)