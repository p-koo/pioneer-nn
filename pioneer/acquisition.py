import torch
from typing import Callable, Union, Optional
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append('./')
from proposer import Proposer

class Acquisition(Proposer):
    """Abstract base class for sequence acquisition.
    
    All acquisition classes should inherit from this class and implement
    the select method.
    """
    def __call__(self, x):
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
    >>> x, idx = Random(target_size=32)
    """
    def __init__(self, target_size, seed=None):
        self.target_size = target_size
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert x.shape[0] >= self.target_size, "Target size is larger than the number of sequences"
        N = x.shape[0]
        idx = torch.randperm(N)[:self.target_size]
        return x[idx], idx


class ScoreAcquisition(Acquisition):
    """Acquisition that selects sequences with highest uncertainty scores.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select
    scorer : Callable
        Function that scores sequences
        
    Examples
    --------
    >>> # Using uncertainty scores from a model wrapper
    >>> acq = ScoreAcquisition(target_size=32, scorer=model.uncertainty)
    >>> x, idx = acq.select(sequences)
    >>> # Using Y scores from a model wrapper
    >>> acq = ScoreAcquisition(target_size=32, scorer=model.predict)
    >>> x, idx = acq.select(sequences)
    """
    def __init__(self, target_size: int, scorer: Callable, batch_size: int = 32, device: Union[str,None] = None):
        self.target_size = target_size
        self.scorer = scorer
        self.batch_size = batch_size
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select sequences with highest scores.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected sequences and their indices
        """
        assert x.shape[0] >= self.target_size, "Target size is larger than the number of sequences"
        if x.shape[0] < self.batch_size:
            scores = self.scorer(x.to(self.device)).squeeze()
            _, idx = torch.sort(scores, descending=True)
            idx = idx[:self.target_size]
            return x[idx], idx
        else:
            #Make data loader
            dataset = TensorDataset(x)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Get uncertainty scores in batches
            scores = []
            for batch in dataloader:
                with torch.no_grad():
                    scores.append(self.scorer(batch[0].to(self.device)).cpu())
            scores = torch.cat(scores, dim=0).squeeze()
            
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
    >>> x, idx = LCMDAcquisition(target_size=32, models=models, x_train=x_train, y_train=y_train)
    """
    def __init__(self, target_size, models, x_train, y_train, base_kernel='grad', 
                 kernel_transforms=[('rp', [512])], sel_with_train=False):
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
        from bmdal_reg.bmdal.algorithms import select_batch
        
        self.target_size = target_size
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.base_kernel = base_kernel
        self.kernel_transforms = kernel_transforms
        self.sel_with_train = sel_with_train
        self.TensorFeatureData = TensorFeatureData
        self.select_batch = select_batch

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        train_data = self.TensorFeatureData(torch.tensor(self.x_train))
        pool_data = self.TensorFeatureData(torch.tensor(x))
        idx, _ = self.select_batch(batch_size=self.target_size, models=self.models, 
                                data={'train': train_data, 'pool': pool_data}, y_train=self.y_train,
                                selection_method=self.selection_method, sel_with_train=self.sel_with_train,
                                base_kernel=self.base_kernel, kernel_transforms=self.kernel_transforms) 

        return x[idx], idx
    

class PoolBasedAcquisition(Acquisition):
    """Acquisition that selects sequences from a pool.
    
    Attributes:
        selector: Acquisition or callable function for selecting sequences from pool
        pool: Pool sequences in one-hot encoding of shape (N, A, L)
        prior_idx: Optional tensor of indices mapping pool sequences to training data
        check_prior_idx: Whether to verify prior indices match training data
    """
    def __init__(self, selector:Union[Acquisition,Callable[[torch.Tensor],torch.Tensor]], 
                 pool:torch.Tensor,
                 prior_idx:Optional[torch.Tensor]=None, 
                 check_prior_idx:bool=True) -> None:
        self.selector = selector
        self.prior_idx = prior_idx
        self.pool = pool
        self.check_prior_idx=check_prior_idx
    
    
    def clean_pool_idx(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean pool using indices.
        
        Removes training sequences from the pool by filtering out sequences with indices
        matching the prior_idx attribute.
        
        Parameters
        ----------
        x : torch.Tensor
            Training sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - pool_idxs: Indices of sequences in pool that are not in training data
            - pool: Pool sequences excluding those in training data, shape (M, A, L) where M is
              the number of remaining sequences after filtering
        
        Raises
        ------
        ValueError
            If prior_idx does not correctly map pool sequences to training sequences
        """
        if self.check_prior_idx:
            try:
                prior_sort_mask = torch.argsort(self.prior_idx)
                x_grab = self.pool[self.prior_idx]
                assert (x[prior_sort_mask]==x_grab[prior_sort_mask]).all(), 'Prior IDX doesnt match X_train'
            except:
                bxb_mat = (x[prior_sort_mask][:,None,:,:]==x_grab[prior_sort_mask][None,:,:,:]).all(axis=-1).all(axis=-1)
                print(f'FEATHER All pristines appear in subselected pool {bxb_mat.any(axis=-1).all()}')
                print(f'FEATHER All subselected pool appear in pristines {bxb_mat.any(axis=0).all()}')
                pristine_idx, subselected_match_idx = torch.where(bxb_mat)
                print(f'FEATHER {pristine_idx=}')
                print(f'FEATHER {subselected_match_idx=}')
                raise ValueError('Prior IDX doesnt match X_train')

        #need to grab pool inverted
        rm_idx = torch.ones(len(self.pool), dtype=torch.bool)
        rm_idx[self.prior_idx] = False
        pool_idxs = torch.where(rm_idx)[0]
        pool = self.pool[pool_idxs]  
        return(pool_idxs,pool)
    
    def __call__(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select sequences from pool.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            (Note: This parameter is unused but kept for interface consistency)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences from pool of shape (target_size, A, L)
            - Indices of selected sequences in original pool
        """
        if self.prior_idx is not None:
            pool_idxs,clean_pool = self.clean_pool_idx(x)
        else:
            clean_pool = x  

        batch = self.selector(clean_pool)

        if self.prior_idx is not None:
            self.selected_idx_in_pool = pool_idxs[batch[1]]
            batch = (batch[0], 
                      pool_idxs[batch[1]])

        return(batch)
