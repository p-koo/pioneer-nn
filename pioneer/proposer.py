
import torch

class Proposer:
    """Abstract base class for sequence proposers.
    
    All proposer classes should inherit from this class and implement
    the __call__ method to propose new sequences based on input sequences.
    This can include generating mutations, combining sequences, or selecting
    promising candidates.
    """
    def __call__(self, x):
        """Propose a batch of sequences.
        
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
            Proposed sequences with same shape as input
        """
        pass

class SequentialProposer(Proposer):
    """Proposer that applies multiple proposers in sequence.
    
    This class takes a list of proposers and applies them sequentially to the input sequences.
    Each proposer's output becomes the input to the next proposer in the list. This allows
    for composing multiple sequence modification strategies, such as applying random mutations
    followed by guided mutations.
    
    Parameters
    ----------
    proposer_list : list[Proposer]
        List of proposer instances to apply in sequence. Can include any combination
        of Generator or Acquisition proposers.
        
    Examples
    --------
    >>> # Apply random mutations followed by guided mutations
    >>> g1 = MutationGenerator(mut_rate=0.1)
    >>> g2 = GuidedMutagenesisGenerator(attr_method, mut_rate=0.2)
    >>> seq_proposer = SequentialProposer([g1, g2])
    >>> proposed_seqs = seq_proposer(sequences)
    """
    def __init__(self, proposer_list: list[Proposer]):
        self.proposer_list = proposer_list
        

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply proposers sequentially to input sequences.
        
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
            Sequences after applying all proposers in sequence,
            with same shape as input (N, A, L). Each sequence has been
            modified by each proposer in order.
        """
        # Apply each proposer in sequence
        x_mut = x.clone()
        for proposer in self.proposer_list:
            x_mut = proposer(x_mut)
        return x_mut


class MultiProposer(Proposer):
    """Proposer that applies multiple proposers in parallel and combines results.
    
    This class takes a list of proposers and applies them independently to the input sequences.
    The outputs from all proposers are concatenated along the batch dimension. This allows
    for exploring multiple sequence modification strategies simultaneously, such as combining
    random and guided mutations into a single candidate pool.
    
    Parameters
    ----------
    proposer_list : list[Proposer]
        List of proposer instances to apply in parallel. Can include any combination
        of Generator or Acquisition proposers.
        
    Examples
    --------
    >>> # Generate both random and guided mutations
    >>> g1 = MutationGenerator(mut_rate=0.1)
    >>> g2 = GuidedMutagenesisGenerator(attr_method, mut_rate=0.2)
    >>> multi_proposer = MultiProposer([g1, g2])
    >>> proposed_seqs = multi_proposer(sequences)  # 2x batch size
    """
    def __init__(self, proposer_list: list[Proposer]):
        self.proposer_list = proposer_list


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply proposers in parallel and combine results.
        
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
            Combined sequences from all proposers,
            shape (N * n_proposers, A, L) where n_proposers
            is the number of proposers in proposer_list. The first N
            sequences are from the first proposer, the next N from
            the second proposer, and so on.
        """
        # Get shape parameters
        N, A, L = x.shape
        
        # Apply each proposer to the original input
        outputs = []
        for proposer in self.proposer_list:
            outputs.append(proposer(x))
            
        # Combine results by concatenating along batch dimension
        x_mut = torch.cat(outputs, dim=0)
        
        return x_mut