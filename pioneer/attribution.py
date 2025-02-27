import torch
from torch.utils.data import DataLoader, TensorDataset


class AttributionMethod:
    """Abstract base class for sequence attribution methods.
    
    All attribution classes should inherit from this class and implement
    the attribute method.
    """
    def __init__(self, model):
        self.model = model
        # Set device based on model
        self.device = next(model.parameters()).device

    def attribute(self, x, batch_size=32):
        """Calculate attribution scores for input sequences.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Attribution scores with same shape as input
        """
        raise NotImplementedError


class UncertaintySaliency(AttributionMethod):
    """Attribution method that calculates gradients of uncertainty w.r.t. inputs.
    
    Args:
        model: ModelWrapper instance with uncertainty estimation capability
        
    Example:
        >>> model = ModelWrapper(base_model, predictor, uncertainty_method)
        >>> attr = UncertaintyAttribution(model)
        >>> scores = attr.attribute(sequences)
    """
    def attribute(self, x, batch_size=32):
        """Calculate uncertainty attribution scores.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Attribution scores of shape (N, A, L)
        """
        # Process sequences in batches
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        attr_scores = []
        
        for batch in loader:
            # Enable gradient tracking for inputs
            x_batch = batch[0].clone().requires_grad_(True)
            
            # Calculate uncertainty and backpropagate
            uncertainty = self.model.uncertainty(x_batch)
            uncertainty.sum().backward()
            
            # Store gradients for this batch
            attr_scores.append(x_batch.grad.detach().clone())
            
        # Combine all batch results
        return torch.cat(attr_scores, dim=0)


class ActivitySaliency(AttributionMethod):
    """Attribution method that calculates gradients of predictions w.r.t. inputs.
    
    Args:
        model: ModelWrapper instance with prediction capability
        
    Example:
        >>> model = ModelWrapper(base_model, predictor)
        >>> attr = ActivityAttribution(model)
        >>> scores = attr.attribute(sequences)
    """
    def attribute(self, x, batch_size=32):
        """Calculate prediction attribution scores.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Attribution scores of shape (N, A, L)
        """
        # Process sequences in batches
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        attr_scores = []
        
        for batch in loader:
            # Enable gradient tracking for inputs
            x_batch = batch[0].clone().requires_grad_(True)
            
            # Calculate predictions and backpropagate
            predictions = self.model.predict(x_batch)
            predictions.sum().backward()
            
            # Store gradients for this batch
            attr_scores.append(x_batch.grad.detach().clone())
            
        # Combine all batch results
        return torch.cat(attr_scores, dim=0)
    
    