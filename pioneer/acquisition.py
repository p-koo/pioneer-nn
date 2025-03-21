import torch
from typing import Callable, Union, Optional
from torch.utils.data import TensorDataset, DataLoader
from pioneer.proposer import Proposer

class Acquisition(Proposer):
    """Abstract base class for sequence acquisition.
    
    All acquisition classes should inherit from this class and implement
    the __call__ method to select promising sequences from a pool of candidates.
    """
    def __call__(self, x):
        """Select sequences from input batch.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (M, A, L)
            - Selection indices of shape (M,)
            where M is determined by the acquisition strategy
        """
        pass


class RandomAcquisition(Acquisition):
    """Acquisition that randomly samples sequences from input.
    
    This class implements random selection of sequences from a pool of candidates.
    It can be used as a baseline acquisition strategy or for exploration.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select from the input pool
    seed : int, optional
        Random seed for reproducible selection, by default None
        
    Examples
    --------
    >>> # Select 32 random sequences
    >>> acquisition = RandomAcquisition(target_size=32, seed=42)
    >>> selected_seqs, indices = acquisition(sequences)
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
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (target_size, A, L)
            - Selection indices of shape (target_size,)
            
        Raises
        ------
        AssertionError
            If target_size is larger than the number of input sequences
        """
        assert x.shape[0] >= self.target_size, "Target size is larger than the number of sequences"
        N = x.shape[0]
        idx = torch.randperm(N)[:self.target_size]
        return x[idx], idx


class ScoreAcquisition(Acquisition):
    """Acquisition that selects sequences with highest scores.
    
    This class implements score-based selection of sequences. It uses a provided
    scoring function (e.g. model predictions or uncertainty estimates) to rank
    sequences and select the top scoring ones.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select from the input pool
    scorer : Callable
        Function that takes sequences and returns scores.
        Should output a tensor of shape (N,) where N is batch size.
    batch_size : int, optional
        Batch size for processing sequences through scorer, by default 32
    device : str, optional
        Device to use for computations ('cuda' or 'cpu').
        If None, automatically selects GPU if available, by default None
        
    Examples
    --------
    >>> # Select sequences with highest uncertainty
    >>> acq = ScoreAcquisition(
    ...     target_size=32,
    ...     scorer=model.uncertainty,
    ...     batch_size=64
    ... )
    >>> selected_seqs, indices = acq(sequences)
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
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (target_size, A, L)
            - Selection indices of shape (target_size,)
            Both sorted by descending score
            
        Raises
        ------
        AssertionError
            If target_size is larger than the number of input sequences
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
    
    This class implements the LCMD selection strategy which aims to maximize diversity
    in feature space while considering the training data distribution. It extracts
    features using provided models and selects sequences that are maximally diverse
    according to kernel-based distance metrics.
    
    Parameters
    ----------
    target_size : int
        Number of sequences to select from the input pool
    models : list[torch.nn.Module]
        List of models used for feature extraction.
        Can be an ensemble or different architectures.
    x_train : torch.Tensor
        Training data used to center the selection,
        shape (N_train, A, L)
    y_train : torch.Tensor
        Training labels corresponding to x_train,
        shape (N_train,)
    base_kernel : str, optional
        Type of kernel for feature extraction:
        - 'grad': Gradient-based features
        - 'last_layer': Features from final layer
        By default 'grad'
    kernel_transforms : list[tuple], optional
        List of kernel transformations to apply.
        Each tuple contains (transform_type, transform_params).
        Default is [('rp', [512])] for random projection.
    sel_with_train : bool, optional
        Whether to include training data in selection pool.
        If True, may select sequences from training set.
        By default False
        
    Examples
    --------
    >>> # Select diverse sequences using gradient features
    >>> acquisition = LCMDAcquisition(
    ...     target_size=32,
    ...     models=ensemble_models,
    ...     x_train=train_sequences,
    ...     y_train=train_labels,
    ...     base_kernel='grad',
    ...     kernel_transforms=[('rp', [1024])]
    ... )
    >>> selected_seqs, indices = acquisition(sequences)
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
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - Selected sequences of shape (target_size, A, L)
            - Selection indices of shape (target_size,)
            Both chosen to maximize diversity in feature space
        """

        train_data = self.TensorFeatureData(torch.tensor(self.x_train))
        pool_data = self.TensorFeatureData(torch.tensor(x))
        idx, _ = self.select_batch(batch_size=self.target_size, models=self.models, 
                                data={'train': train_data, 'pool': pool_data}, y_train=self.y_train,
                                selection_method=self.selection_method, sel_with_train=self.sel_with_train,
                                base_kernel=self.base_kernel, kernel_transforms=self.kernel_transforms) 

        return x[idx], idx