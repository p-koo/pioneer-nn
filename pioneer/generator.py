import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Union, Optional

from pioneer.acquisition import Acquisition


class Generator():
    """Abstract base class for sequence generators.
    
    All generator classes should inherit from this class and implement
    the __call__ method to generate modified sequences from input sequences.
    """
    def __call__(self, x):
        """Generate modified sequences from input sequences.
        
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
            Modified sequences with same shape as input
        """
        pass

class RandomGenerator(Generator):
    """Generator that creates random sequences based on nucleotide probabilities.
    
    This class generates random sequences by sampling nucleotides according to specified
    probabilities. Can optionally restrict mutations to a specific sequence window.
    
    Parameters
    ----------
    prob : list[float], optional
        Probabilities for each nucleotide, must sum to 1.0, by default [0.25, 0.25, 0.25, 0.25]
    seed : int, optional
        Random seed for reproducible sequence generation, by default None
    mut_window : tuple[int, int], optional
        Start and end positions defining mutation window.
        If None, generates random sequences for entire length, by default None
        
    Examples
    --------
    >>> # Generate sequences with biased nucleotide probabilities
    >>> gen = RandomGenerator(prob=[0.3, 0.2, 0.2, 0.3])
    >>> random_seqs = gen(sequences)
    
    >>> # Generate sequences with mutations only in positions 10-20
    >>> gen = RandomGenerator(mut_window=(10,20))
    >>> random_seqs = gen(sequences)
    """
    def __init__(self, prob: Union[list[float], None] = None, seed: Union[int, None] = None, 
                 mut_window: Union[tuple[int, int], None] = None):
        # Set nucleotide probabilities (default to uniform)
        if prob is None:
            self.prob = torch.tensor([0.25] * 4)
        else:
            prob = torch.tensor(prob)
            assert prob.size(0) == 4 and torch.abs(torch.sum(prob) - 1.0) < 1e-6
            self.prob = prob
        if seed is not None:
            torch.manual_seed(seed)

        self.mut_window = mut_window

    def __call__(self, x):
        """Generate random sequences using specified probabilities.
        
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
            Random sequences with same shape as input. Within the mutation window
            (if specified), nucleotides are randomly sampled according to self.prob.
            Outside the window, sequences remain unchanged.
        """
        if self.mut_window is None:
            start = 0
            end = x.shape[2]
        else:
            start = self.mut_window[0]
            end = self.mut_window[1]

        N, A, L = x.shape
        mutable_L = end - start
        nt_idx = torch.multinomial(self.prob, N * mutable_L, replacement=True)
        mutatations = torch.zeros(N*mutable_L, A, device=x.device)
        mutatations[torch.arange(N*mutable_L), nt_idx] = 1
        mutatations = mutatations.view(N, mutable_L, A)
        mutatations = mutatations.transpose(1,2)

        x = x.clone()
        x[:, :, start:end] = mutatations
        return x


class MutagenesisGenerator(Generator):
    """Generator that randomly mutates sequences within a specified window.
    
    This class implements random mutagenesis by selecting positions at random and
    mutating them to different nucleotides. The mutation rate controls what fraction
    of positions are mutated.
    
    Parameters
    ----------
    mut_rate : float
        Mutation rate between 0 and 1, specifying fraction of positions to mutate
    mut_window : tuple[int, int], optional
        Start and end positions defining mutation window.
        If None, mutations can occur anywhere in sequence, by default None
    seed : int, optional
        Random seed for reproducible mutations, by default None
    batch_size : int, optional
        Batch size for processing sequences. Lower values use less GPU memory
        but may be slower, by default 32
            
    Examples
    --------
    >>> # Mutate 10% of positions randomly
    >>> gen = MutagenesisGenerator(mut_rate=0.1)
    >>> mutated = gen(sequences)
    
    >>> # Mutate 20% of positions between indices 10-20
    >>> gen = MutagenesisGenerator(mut_rate=0.2, mut_window=(10,20))
    >>> mutated = gen(sequences)
    """
    def __init__(self, mut_rate: float = 0.1, 
                 mut_window: Union[tuple[int, int], None] = None, 
                 seed: Union[int, None] = None, 
                 batch_size: int = 32):
        assert 0 <= mut_rate <= 1
        self.mut_rate = mut_rate
        self.mut_window = mut_window
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed is not None:
            torch.manual_seed(seed)
        self.batch_size = batch_size

    def __call__(self, x):
        """Generate mutated sequences based on mutation rate.
        
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
            Mutated sequences with same shape as input. Within the mutation window
            (if specified), mut_rate fraction of positions are randomly mutated to
            different nucleotides. Outside the window, sequences remain unchanged.
        """
        N_total, A, L = x.shape

        # Set mutation window
        start = 0 if self.mut_window is None else self.mut_window[0]
        end = L if self.mut_window is None else self.mut_window[1]
        window_size = end - start

        x_mut = []

        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for batch_x, in loader:
            # Convert one-hot to indices
            current_nt_indices = batch_x.argmax(dim=1).to(self.device)
            N = current_nt_indices.shape[0]
            
            mutation_rotation = torch.zeros(N, window_size, device=self.device).float()
            
            # Generate random mutations
            mut_num = int(window_size*self.mut_rate)
            mutation_idx = torch.argsort(torch.rand(N, window_size, device=self.device), dim=1)[:, :mut_num] #random positions to mutate
            batch_idx, _ = torch.meshgrid(torch.arange(N, device=self.device), torch.arange(mut_num, device=self.device), indexing='ij')
            mutation_rotation[batch_idx.flatten(), mutation_idx.flatten()] = torch.randint(1, A, (N*mut_num,), device=self.device).float() # random over alphabet
            
            # Apply mutations and convert back to one-hot
            current_nt_indices[:, start:end] = (current_nt_indices[:, start:end] + mutation_rotation) % A
            batch_mut = F.one_hot(current_nt_indices, num_classes=A).permute(0, 2, 1).detach().cpu()
            x_mut.append(batch_mut)
        
        return torch.cat(x_mut, dim=0).float()


class GuidedMutagenesisGenerator(Generator):
    """Generator that uses attribution scores to guide mutations.
    
    This class implements guided mutagenesis by using attribution scores to determine
    which positions and nucleotides are most important for the model's predictions.
    The temperature parameter controls how strictly the mutations follow the attribution
    scores versus random sampling.
    
    Parameters
    ----------
    attr_method : callable
        Method that computes attribution scores for sequences. Should return scores
        indicating importance of each position and nucleotide.
    mut_rate : float
        Mutation rate between 0 and 1, specifying fraction of positions to mutate
    mut_window : tuple[int, int], optional
        Start and end positions defining mutation window.
        If None, mutations can occur anywhere in sequence, by default None
    temp : float or str, optional
        Temperature for softmax sampling:
        - 'neg_inf' or negative values: deterministic selection of highest scores
        - positive values: sample proportional to scores, higher = more random
        By default -1 (deterministic)
    seed : int, optional
        Random seed for reproducible mutations, by default None
    batch_size : int, optional
        Batch size for processing sequences. Lower values use less GPU memory
        but may be slower, by default 32
    dont_repeat_positions : bool, optional
        If True, prevents mutating the same position multiple times in a sequence,
        by default True
            
    Examples
    --------
    >>> # Deterministic mutations guided by attribution scores
    >>> gen = GuidedMutagenesisGenerator(attr_method, mut_rate=0.1)
    >>> guided_mutations = gen(sequences)
    
    >>> # Probabilistic mutations with temperature=1.0
    >>> gen = GuidedMutagenesisGenerator(attr_method, mut_rate=0.1, temp=1.0)
    >>> guided_mutations = gen(sequences)
    """
    def __init__(self, attr_method: Callable, mut_rate: float = 0.1, 
                 mut_window: Union[tuple[int, int], None] = None, 
                 temp: Union[float, str] = -1, seed: Union[int, None] = None, 
                 batch_size: int = 32,
                 dont_repeat_positions: bool = True):
        assert 0 <= mut_rate <= 1
        self.attr_method = attr_method
        self.mut_rate = mut_rate
        self.mut_window = mut_window
        self.temp = temp
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed is not None:
            torch.manual_seed(seed)
        self.batch_size = batch_size
        self.dont_repeat_positions = dont_repeat_positions

    def __call__(self, x):
        """Generate mutations guided by attribution scores.
        
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
            Mutated sequences with same shape as input. Within the mutation window
            (if specified), positions are selected for mutation based on attribution
            scores and temperature parameter:
            - Low temperature: Mutations target highest attribution scores
            - High temperature: More random selection weighted by scores
            Outside the window, sequences remain unchanged.
        """
        

        # Set mutation window
        start = 0 if self.mut_window is None else self.mut_window[0]
        end = x.shape[2] if self.mut_window is None else self.mut_window[1]
        window_size = end - start

        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        mutations = []
        for batch_x, in loader:
            # Move batch to GPU
            batch_x = batch_x.to(self.device)
            
            # Get shape parameters for this batch
            batch_size, A, L = batch_x.shape
            
            
            n_mutations = torch.floor(torch.tensor(window_size * self.mut_rate)).long()
            
            # Clone input for mutations
            batch_mut = batch_x.clone().detach()
            
            # Track mutated positions
            if self.dont_repeat_positions:
                mutated_positions = torch.zeros((batch_size, window_size), 
                                             dtype=torch.bool, device=self.device)
            
            # Apply mutations one at a time
            for _ in range(n_mutations):
                # Get attribution scores
                attr_map = self.attr_method(batch_x)
                attr_map = attr_map[:, :, start:end]
                
                # Zero out current nucleotide scores and previously mutated positions
                current_nt_mask = batch_mut[:, :, start:end].bool()
                attr_map[current_nt_mask] = torch.tensor(float('-inf'))
                if self.dont_repeat_positions:
                    seq_mask, pos_mask = torch.where(mutated_positions)
                    attr_map[seq_mask, :, pos_mask] = torch.tensor(float('-inf'))
                
                # Apply temperature scaling if specified
                if self.temp > 0:
                    probs = F.softmax(attr_map.reshape(batch_size, -1) / self.temp, dim=1)
                else:
                    reshaped_map = attr_map.reshape(batch_size, -1)
                    p_inds = torch.argmax(reshaped_map, dim=1)
                    probs = torch.zeros_like(reshaped_map, device=self.device)
                    probs[torch.arange(batch_size, device=self.device), p_inds] = 1
                
                # Convert to probabilities and sample mutations
                
                mut_indices = torch.multinomial(probs, 1).squeeze()
                
                # Convert to nucleotide and position indices
                new_nts = mut_indices // attr_map.shape[2]
                mut_positions = mut_indices % attr_map.shape[2]
                
                # Apply single mutation using aligned indices
                batch_idx = torch.arange(batch_size, device=self.device)
                batch_mut[batch_idx, :, mut_positions + start] = 0
                batch_mut[batch_idx[None].T, new_nts[:, None], 
                         mut_positions[None].T + start] = 1
                
                # Update mutated positions tracker
                if self.dont_repeat_positions:
                    mutated_positions[batch_idx, mut_positions] = True
            
            # Move results back to CPU
            mutations.append(batch_mut.cpu())
            
        return torch.cat(mutations, dim=0)
    


class PoolBasedGenerator(Generator):
    """Generator that selects sequences from a pre-defined pool.
    
    This class implements sequence generation by selecting sequences from a pool of
    candidates using an acquisition function or selector. Can optionally exclude
    sequences that are already present in the training data.
    
    Parameters
    ----------
    selector : Union[pioneer.acquisition.Acquisition, Callable[[torch.Tensor], torch.Tensor]]
        Acquisition function or callable that selects sequences from the pool.
        Should return selected sequences and their indices.
    pool : torch.Tensor
        Pool of candidate sequences in one-hot encoding of shape (N, A, L)
    prior_idx : Optional[torch.Tensor], optional
        Indices mapping pool sequences to training data sequences. Used to exclude
        training sequences from selection, by default None
    check_prior_idx : bool, optional
        Whether to verify prior_idx correctly maps pool sequences to training data.
        Only used if prior_idx is provided, by default True
        
    Examples
    --------
    >>> # Select sequences using uncertainty acquisition
    >>> selector = UncertaintyAcquisition(n_select=10)
    >>> gen = PoolBasedGenerator(selector, sequence_pool)
    >>> selected = gen(sequences)
    
    >>> # Select excluding training sequences
    >>> gen = PoolBasedGenerator(selector, sequence_pool, prior_idx=train_idx)
    >>> selected = gen(sequences)
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
        """Clean pool by removing sequences that appear in training data.
        
        Parameters
        ----------
        x : torch.Tensor
            Training sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - pool_idxs: Indices of sequences in pool that are not in training data
            - pool: Filtered pool sequences excluding those in training data
        
        Raises
        ------
        AssertionError
            If prior_idx does not correctly map pool sequences to training sequences
            when check_prior_idx is True
        """
        if self.check_prior_idx:
            prior_sort_mask = torch.argsort(self.prior_idx)
            x_grab = self.pool[self.prior_idx]
            assert (x[prior_sort_mask]==x_grab[prior_sort_mask]).all(), 'Prior IDX doesnt match X_train'

        #need to grab pool inverted
        rm_idx = torch.ones(len(self.pool), dtype=torch.bool)
        rm_idx[self.prior_idx] = False
        pool_idxs = torch.where(rm_idx)[0]
        pool = self.pool[pool_idxs]  
        return(pool_idxs,pool)
    
    def __call__(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select sequences from the pool using the selector function.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            Used to filter pool if prior_idx is provided
            
        Returns
        -------
        torch.Tensor
            Selected sequences from the pool, chosen by the selector function.
            If prior_idx is provided, excludes sequences present in x from selection.
            Shape matches selector output.
        """
        if self.prior_idx is not None:
            pool_idxs,clean_pool = self.clean_pool_idx(x)
        else:
            clean_pool = x  

        selected_x, selected_clean_pool_idx = self.selector(clean_pool)

        if self.prior_idx is not None:
            self.selected_idx_in_pool = pool_idxs[selected_clean_pool_idx]
        else:
            self.selected_idx_in_pool = selected_clean_pool_idx
            

        return(selected_x)
