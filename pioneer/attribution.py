import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

class AttributionMethod:
    """Abstract base class for sequence attribution methods.
    
    All attribution classes should inherit from this class and implement
    the attribute method to calculate attribution scores for input sequences.
    These scores indicate how much each position in the input sequence contributes
    to the model's predictions.

    Parameters
    ----------
    scorer : Callable
        Function that takes input sequences and returns scores/predictions to attribute.
        This could be a model's forward pass or uncertainty estimation function.
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
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
        batch_size : int, optional
            Batch size for processing, by default 32
                
        Returns
        -------
        torch.Tensor
            Attribution scores with same shape as input (N, A, L).
            Higher absolute values indicate positions that more strongly
            influence the model's predictions.
        """
        raise NotImplementedError
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.attribute(x)


class Saliency(AttributionMethod):
    """Attribution method that calculates gradients with respect to inputs.
    
    This class computes attribution scores using gradient-based saliency mapping.
    For each position in the input sequence, it calculates how much a small change
    would affect the model's output by computing the gradient of the output with
    respect to that position. The magnitude of the gradient indicates the position's
    importance.
    
    Parameters
    ----------
    scorer : Callable
        Function that takes input sequences and returns differentiable scores/predictions.
        This could be a model's forward pass for prediction-based attribution or
        an uncertainty estimation function for uncertainty-based attribution.
        
    Examples
    --------
    >>> # Attribution based on model predictions
    >>> model = ModelWrapper(base_model, predictor)
    >>> attr = Saliency(model.predict)
    >>> pred_scores = attr(sequences)
    >>>
    >>> # Attribution based on model uncertainty 
    >>> attr = Saliency(model.uncertainty)
    >>> uncert_scores = attr(sequences)
    """
    def __init__(self, scorer: Callable):
        self.scorer = scorer
        
    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate gradient-based attribution scores.
        
        For each position in the input sequences, computes the gradient of the
        scorer's output with respect to that position. The absolute magnitude
        of the gradient indicates how sensitive the output is to changes at
        that position.
        
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
            Attribution scores of shape (N, A, L). Each value represents
            the gradient of the scorer's output with respect to that input position.
            Larger absolute values indicate positions that more strongly affect
            the output when perturbed.
        """
        # Enable gradient tracking for inputs
        x = x.clone().requires_grad_(True)
        
        # Calculate score and backpropagate
        score = self.scorer(x)
        score.sum().backward()
        
        # Store gradients for this batch
        attr_scores = x.grad.detach().clone()
            
        return attr_scores
