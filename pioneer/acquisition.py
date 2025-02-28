import torch

from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch


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
    """Acquisition that selects sequences using LCMD (Linear Centered Maximum Distance) method.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select
    models : list
        List of models for LCMD selection
    x_train : torch.Tensor
        Training data used for selection
    y_train : torch.Tensor
        Training labels used for selection
    device : str, optional
        Device to use for computations, by default 'cuda'
    batch_size : int, optional
        Batch size for computations, by default 100
    base_kernel : str, optional
        Kernel type for selection, by default 'grad'
    kernel_transforms : list, optional
        List of kernel transformations, by default [('rp', [512])]
    sel_with_train : bool, optional
        Whether to include training data in selection, by default False
        
    Examples
    --------
    >>> acq = LCMDAcquisition(target_size=32, models=models, x_train=x_train, y_train=y_train)
    >>> selected_seqs, indices = acq.select(sequences)
    """
    def __init__(self, target_size, models, x_train, y_train, device='cuda', 
                 batch_size=100, base_kernel='grad', 
                 kernel_transforms=[('rp', [512])], sel_with_train=False):
        self.target_size = target_size
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.device = device
        self.batch_size = batch_size
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms
        self.sel_with_train = sel_with_train

    def select(self, x):
        """Select sequences using LCMD method.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected sequences and their indices
        """
        precomp_batch_size=self.external_batch_size #QUIQUINEW NOT USED!!!
        nn_batch_size=self.external_batch_size #QUIQUINEW NOT USED!!!

        train_data = TensorFeatureData(torch.tensor(self.x_train))
        pool_data = TensorFeatureData(torch.tensor(x_pool))


        idx, _ = select_batch(batch_size=self.n_to_get, models=self.models, 
                                data={'train': train_data, 'pool': pool_data}, y_train=self.y_train,
                                selection_method=self.selection_method, sel_with_train=self.sel_with_train,
                                base_kernel=self.base_kernel, kernel_transforms=self.kernel_transforms) # return batch_idxs, results_dict : def select within class BatchSelectorImpl in algorithms.py
        return(idx)
