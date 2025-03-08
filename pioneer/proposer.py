
import torch

class Proposer:
    """Abstract base class for sequence generators.
    
    All Proposer classes should inherit from this class and implement
    the __call__ method.
    """
    def __call__(self, x):
        """Propose a batch of sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
                
        Returns
        -------
        torch.Tensor
            Proposed sequences with same shape as input
        """
        pass

class SequentialProposer(Proposer):
    """Proposer that applies multiple proposers in sequence.
    
    Args:
        generator_list (list[Generator]): List of generators to apply in sequence
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        
    Example:
        >>> g1 = Mutagenesis(mut_rate=0.1)
        >>> g2 = GuidedMutagenesis(attr_method, mut_rate=0.2)
        >>> seq_gen = Sequential([g1, g2])
        >>> mutated = seq_gen.generate(sequences)
    """
    def __init__(self, proposer_list: list[Proposer]):
        self.proposer_list = proposer_list
        

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply generators sequentially to input sequences.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
                
        Returns:
            torch.Tensor: Mutated sequences after applying all generators in sequence,
                with same shape as input
        """
        # Apply each generator in sequence to mutate sequences
        x_mut = x.clone()
        for proposer in self.proposer_list:
            x_mut = proposer(x_mut)
        return x_mut


class MultiProposer(Proposer):
    """Proposer that applies multiple proposers in parallel and combines results.
    
    Args:
        generator_list (list[Generator]): List of generators to apply in parallel
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        
    Example:
        >>> g1 = Mutagenesis(mut_rate=0.1)
        >>> g2 = GuidedMutagenesis(attr_method, mut_rate=0.2)
        >>> multi_gen = MultiGenerator([g1, g2])
        >>> mutated = multi_gen.generate(sequences)  # 2x batch size
    """
    def __init__(self, proposer_list: list[Proposer]):
        self.proposer_list = proposer_list


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply generators in parallel and combine results.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
                
        Returns:
            torch.Tensor: Combined mutated sequences from all generators,
                shape (N * n_generators, A, L)
        """
        # Get shape parameters
        N, A, L = x.shape
        
        # Apply each generator to the original input
        outputs = []
        for proposer in self.proposer_list:
            outputs.append(proposer(x))
            
        # Combine results by concatenating along batch dimension
        x_mut = torch.cat(outputs, dim=0)
        
        return x_mut