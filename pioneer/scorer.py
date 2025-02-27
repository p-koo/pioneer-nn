from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from typing import Union

class Scorer(metaclass=ABCMeta):
    """Abstract base class for computing model predictions/scores in batches.
    
    Attributes:
        model: The model to get predictions from.
        batch_size: Size of batches to process at once.
    """
    def __init__(self, model, batch_size:int) -> None:
        self.model = model
        self.batch_size = batch_size

    def __call__(self,x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Score input x in batches.
        
        Args:
            x: Input data to score, can be numpy array or PyTorch tensor.
               Must be of shape (Data, alphabet, length) where:
               - Data is the number of sequences
               - alphabet is the size of the sequence alphabet (e.g. 4 for DNA)
               - length is the sequence length
            
        Returns:
            torch.Tensor of shape (Data, 1) containing scores for each input sample.
                Where:
                - Data is the number of sequences
        """
        batch_scores = []
        for batch in torch.utils.data.DataLoader(x,batch_size=self.batch_size):
            batch_scores.append(self.score(batch))
        return(torch.cat(batch_scores))
    
    @abstractmethod
    def score(self,x: torch.Tensor) -> torch.Tensor:
        """Score a single batch of inputs.
        
        Args:
            x: Batch of inputs to score as a PyTorch tensor.
            
        Returns:
            torch.Tensor containing scores for the batch.
        """
        pass
    

class UncertiantyScorer(Scorer):
    """Scorer that computes model uncertainty scores.
    
    This scorer returns uncertainty estimates from models that support uncertainty
    quantification.
    """
    def score(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty scores for a batch.
        
        Args:
            x: Batch of inputs as numpy array or PyTorch tensor.
                Must be of shape (Data, alphabet, length) where:
                - Data is the number of sequences
                - alphabet is the size of the sequence alphabet (e.g. 4 for DNA)
                - length is the sequence length
            
        Returns:
            torch.Tensor containing uncertainty scores for each input sample.
                Must be of shape (Data, 1) where:
                - Data is the number of sequences
        """
        return(self.model.uncertainty(x.to(next(self.model.parameters()).device), keepgrad=True))
            
class ActivityScorer(Scorer):
    """Scorer that returns raw model predictions.
    
    This scorer simply returns the direct outputs from the model without any
    additional processing.
    """
    def score(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get model predictions for a batch.
        
        Args:
            x: Batch of inputs as numpy array or PyTorch tensor.
                Must be of shape (Data, alphabet, length) where:
                - Data is the number of sequences
                - alphabet is the size of the sequence alphabet (e.g. 4 for DNA)
                - length is the sequence length
            
        Returns:
            torch.Tensor containing raw model predictions.
                Must be of shape (Data, 1) where:
                - Data is the number of sequences
        """
        return(self.model(x.to(next(self.model.parameters()).device)))


class SaliencyAttribution(Scorer):
    """Attribution method that computes input saliency maps via gradients.
    
    Computes feature importance by calculating gradients of the model output
    with respect to the input features.
    """
    def __call__(self,x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute saliency maps for input x.
        
        Args:
            x: Input data to compute saliency maps for, can be numpy array or PyTorch tensor.
                Must be of shape (Data, alphabet, length) where:
                - Data is the number of sequences
                - alphabet is the size of the sequence alphabet (e.g. 4 for DNA)
                - length is the sequence length
            
        Returns:
            numpy.ndarray containing saliency scores for each input feature.
                Must be of shape (Data, alphabet, length) where:
                - Data is the number of sequences
                - alphabet is the size of the sequence alphabet (e.g. 4 for DNA)
        """
        batch_sails = []
        for batch in torch.utils.data.DataLoader(x,batch_size=self.scorer.batch_size):
            x = batch.float().requires_grad_()
            x.retain_grad()

            unc_all = self.scorer.score(x)
            unc_all.sum().backward()
            
            unc_sail = x.grad.data.cpu().numpy()
            batch_sails.append(unc_sail)
        
        # Convert saliencies to a numpy array and cut to the mutable region
        unc_sail = np.concatenate(batch_sails)
        return(unc_sail)
    

    