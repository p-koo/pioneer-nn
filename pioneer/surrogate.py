import torch
from torch.utils.data import DataLoader, TensorDataset


class ModelWrapper:
    """Wrapper for models with flexible prediction and uncertainty estimation.
    
    Args:
        model: PyTorch model or list of models
        predictor: Method for generating predictions
        uncertainty_method (optional): Method for uncertainty estimation. 
            Defaults to None.
            
    Example:
        >>> # Single model with MC Dropout
        >>> model = MyModel.load_from_checkpoint("model.ckpt")
        >>> custom_model = CustomModel(
        ...     model,
        ...     predictor=ProfilePredictor(reduction=profile_sum),
        ...     uncertainty_method=MCDropout(n_samples=20)
        ... )
    """
    def __init__(self, model, predictor, uncertainty_method=None):
        self.model = model
        self.predictor = predictor
        self.uncertainty_method = uncertainty_method
        
        # Set device based on model type
        if isinstance(model, list):
            self.device = next(model[0].parameters()).device
        else:
            self.device = next(model.parameters()).device

    def predict(self, x, batch_size=32):
        """Generate predictions for input sequences.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Model predictions with shape depending on predictor type
        """
        if isinstance(self.model, list):
            # For ensemble, average predictions from all models
            preds = []
            for model in self.model:
                pred = self.predictor.predict(model, x, batch_size)
                preds.append(pred)
            return torch.stack(preds).mean(dim=0)
        else:
            return self.predictor.predict(self.model, x, batch_size)

    def uncertainty(self, x, batch_size=32):
        """Generate uncertainty estimates for input sequences.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Uncertainty scores for each sequence
        """
        if self.uncertainty_method is None:
            raise ValueError("No uncertainty method specified")
        return self.uncertainty_method.estimate(self.model, x, batch_size)
