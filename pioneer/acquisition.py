import torch


class Acquisition:
    """Abstract base class for sequence acquisition.
    
    All acquisition classes should inherit from this class and implement
    the select method.
    """
    def select(self, x):
        """Select sequences from input batch.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected sequences and their indices
        """
        pass


class RandomAcquisition(Acquisition):
    """Acquisition that randomly samples sequences from input.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Examples
    --------
    >>> acq = Random(target_size=32)
    >>> selected_seqs, indices = acq.select(sequences)
    """
    def __init__(self, target_size, seed=None):
        self.target_size = target_size
        if seed is not None:
            torch.manual_seed(seed)

    def select(self, x):
        """Randomly select sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected sequences and their indices
        """
        N = x.shape[0]
        idx = torch.randperm(N)[:self.target_size]
        return x[idx], idx


class UncertaintyAcquisition(Acquisition):
    """Acquisition that selects sequences with highest uncertainty scores.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select
    surrogate_model : ModelWrapper
        Model that provides uncertainty scores
        
    Examples
    --------
    >>> acq = Uncertainty(target_size=32, model=uncertainty_model)
    >>> selected_seqs, indices = acq.select(sequences, batch_size=512)
    """
    def __init__(self, target_size, surrogate_model):
        self.target_size = target_size
        self.surrogate_model = surrogate_model

    def select(self, x, batch_size=32):
        """Select sequences with highest uncertainty scores.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for uncertainty computation, by default 32
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected sequences and their indices
        """
        # Get uncertainty scores in batches
        scores = self.surrogate_model.uncertainty(x, batch_size=batch_size)
        
        # Get indices of top uncertainty scores
        _, idx = torch.sort(scores, descending=True)
        idx = idx[:self.target_size]
        
        return x[idx], idx
    

class LCMDAcquisition(Acquisition):

    def __init__(self):
        super().__init__()

    def select(x, batch_size):

        return x[idx], idx


