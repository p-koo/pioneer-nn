import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

class AttributionMethod:
    """Abstract base class for sequence attribution methods.
    
    All attribution classes should inherit from this class and implement
    the attribute method to calculate attribution scores for input sequences.

    Parameters
    ----------
    scorer : Callable
        Function that takes input sequences and returns scores/predictions to attribute
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
            Attribution scores with same shape as input (N, A, L), indicating
            the importance/contribution of each position and nucleotide
        """
        raise NotImplementedError
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.attribute(x)


class Saliency(AttributionMethod):
    """Attribution method that calculates gradients with respect to inputs.
    
    This class computes attribution scores by calculating the gradients of the scorer's
    output with respect to each input position. The gradient magnitude indicates
    how much changing that position would affect the output.
    
    Parameters
    ----------
    scorer : Callable
        Function that takes input sequences and returns differentiable scores/predictions
        
    Examples
    --------
    >>> model = ModelWrapper(base_model, predictor)
    >>> attr = Saliency(model)
    >>> scores = attr.attribute(sequences)  # Returns gradient-based importance scores
    >>> scores = attr(sequences)  # Alternative call syntax
    """
    def __init__(self, scorer: Callable):
        self.scorer = scorer
        
    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate gradient-based attribution scores.
        
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
            Attribution scores of shape (N, A, L), where each value represents
            the gradient of the output with respect to that input position
        """
        # Enable gradient tracking for inputs
        x = x.clone().requires_grad_(True)
        
        # Calculate score and backpropagate
        score = self.scorer(x)
        score.sum().backward()
        
        # Store gradients for this batch
        attr_scores = x.grad.detach().clone()
            
        return attr_scores

    