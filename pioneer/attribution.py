import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

class AttributionMethod:
    """Abstract base class for sequence attribution methods.
    
    All attribution classes should inherit from this class and implement
    the attribute method.

    Parameters
    ----------
    scorer : Callable
        The scorer to compute attributions for
    """
    def __init__(self, scorer: Callable):
        self.scorer = scorer

    def attribute(self, x, batch_size=32):
        """Calculate attribution scores for input sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing, by default 32
                
        Returns
        -------
        torch.Tensor
            Attribution scores with same shape as input
        """
        raise NotImplementedError


class Saliency(AttributionMethod):
    """Attribution method that calculates gradients w.r.t. inputs.
    
    This class computes attribution scores by calculating the gradients of the model's
    predictions with respect to the input sequences.
    
    Parameters
    ----------
    model : ModelWrapper
        ModelWrapper instance with uncertainty estimation capability
        
    Examples
    --------
    >>> model = ModelWrapper(base_model, predictor, uncertainty_method)
    >>> attr = Saliency(model)
    >>> scores = attr.attribute(sequences)
    """
    def __init__(self, scorer: Callable):
        self.scorer = scorer
        
    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate attribution scores.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing, by default 32
                
        Returns
        -------
        torch.Tensor
            Attribution scores of shape (N, A, L)
        """
        # Enable gradient tracking for inputs
        x = x.clone().requires_grad_(True)
        
        # Calculate score and backpropagate
        score = self.scorer(x)
        score.sum().backward()
        
        # Store gradients for this batch
        attr_scores = x.grad.detach().clone()
            
        return attr_scores

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.attribute(x)