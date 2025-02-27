import torch

class ModelWrapper:
    """Model wrapper for predictions and uncertainty estimation.
    
    Args:
        model: Model or list of models
        predictor: Prediction method
        uncertainty_method: Uncertainty estimation method
        
    Example:
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
        
        Args:
            x: Input sequences
            batch_size: Batch size for processing
            
        Returns:
            Model predictions
        """
        if isinstance(self.model, list):
            preds = [self.predictor.predict(m, x, batch_size) for m in self.model]
            return torch.stack(preds).mean(dim=0)
        return self.predictor.predict(self.model, x, batch_size)

    def uncertainty(self, x, batch_size=32):
        """Generate uncertainty estimates.
        
        Args:
            x: Input sequences
            batch_size: Batch size for processing
            
        Returns:
            Uncertainty scores
        """
        if self.uncertainty_method is None:
            raise ValueError("No uncertainty method specified")
        return self.uncertainty_method.estimate(self.model, x, batch_size)