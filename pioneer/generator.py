import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class Generator:
    """Abstract base class for sequence generators.
    
    All generator classes should inherit from this class and implement
    the generate method.
    """
    def generate(self, x):
        """Generate modified sequences from input sequences.
        
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
            Modified sequences with same shape as input
        """
        pass


class Random(Generator):
    """Generator that creates random sequences based on nucleotide probabilities.
    
    Parameters
    ----------
    prob : list[float], optional
        Probabilities for each nucleotide, by default [0.25, 0.25, 0.25, 0.25]
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Examples
    --------
    >>> gen = Random(prob=[0.3, 0.2, 0.2, 0.3])
    >>> random_seqs = gen.generate(sequences)
    """
    def __init__(self, prob=None, seed=None):
        # Set nucleotide probabilities (default to uniform)
        if prob is None:
            self.prob = torch.tensor([0.25] * 4)
        else:
            prob = torch.tensor(prob)
            assert prob.size(0) == 4 and torch.abs(torch.sum(prob) - 1.0) < 1e-6
            self.prob = prob
        if seed is not None:
            torch.manual_seed(seed)

    def generate(self, x):
        """Generate random sequences using specified probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        torch.Tensor
            Random sequences with same shape as input
        """
        N, A, L = x.shape
        return torch.multinomial(self.prob, N * L, replacement=True).view(N, A, L)


class Mutagenesis(Generator):
    """Generator that randomly mutates sequences within a specified window.
    
    Parameters
    ----------
    mut_rate : float
        Mutation rate between 0 and 1
    mut_window : tuple[int, int], optional
        Start and end positions for mutation window.
        If None, mutates entire sequence, by default None
    seed : int, optional
        Random seed for reproducibility, by default None
            
    Examples
    --------
    >>> gen = Mutagenesis(mut_rate=0.1, mut_window=(10, 20))
    >>> mutated = gen.generate(sequences)
    """
    def __init__(self, mut_rate=0.1, mut_window=None, seed=None):
        assert 0 <= mut_rate <= 1
        self.mut_rate = mut_rate
        self.mut_window = mut_window
        if seed is not None:
            torch.manual_seed(seed)

    def generate(self, x):
        """Generate mutated sequences based on mutation rate.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
            
        Returns
        -------
        torch.Tensor
            Mutated sequences with same shape as input
        """
        N, A, L = x.shape

        # Convert one-hot to indices
        indices = x.argmax(dim=1)
        
        # Set mutation window
        start = 0 if self.mut_window is None else self.mut_window[0]
        end = L if self.mut_window is None else self.mut_window[1]
        window_size = end - start
        window_mutations = torch.zeros(N, window_size)
        
        # Generate random mutations
        mask = torch.rand(N, window_size) < self.mut_rate
        window_mutations[mask] = torch.randint(1, A, (mask.sum(),))
        
        # Apply mutations and convert back to one-hot
        indices[:, start:end] = (indices[:, start:end] + window_mutations) % A
        x_mut = F.one_hot(indices, num_classes=A).permute(0, 2, 1)
        
        return x_mut


class GuidedMutagenesis(Generator):
    """Generator that uses attribution scores to guide mutations.
    
    Parameters
    ----------
    attr_method : callable
        Method that returns attribution scores for sequences
    mut_rate : float
        Mutation rate between 0 and 1
    mut_window : tuple[int, int], optional
        Start and end positions for mutation window.
        If None, mutates entire sequence, by default None
    temp : float or str, optional
        Temperature for softmax. Use 'neg_inf' for deterministic
        selection or positive float for sampling, by default -1
    seed : int, optional
        Random seed for reproducibility, by default None
            
    Examples
    --------
    >>> gen = GuidedMutagenesis(attr_method, mut_rate=0.1, temp=1.0)
    >>> guided_mutations = gen.generate(sequences)
    """
    def __init__(self, attr_method, mut_rate=0.1, mut_window=None, temp=-1, seed=None):
        assert 0 <= mut_rate <= 1
        self.attr_method = attr_method
        self.mut_rate = mut_rate
        self.mut_window = mut_window
        self.temp = temp
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed is not None:
            torch.manual_seed(seed)

    def generate(self, x, batch_size=32):
        """Generate mutations guided by attribution scores.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
                
        Returns
        -------
        torch.Tensor
            Mutated sequences with same shape as input
        """
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        mutations = []
        for batch_x, in loader:
            # Move batch to GPU
            batch_x = batch_x.to(self.device)
            
            # Get shape parameters for this batch
            batch_size, A, L = batch_x.shape
            
            # Set mutation window
            start = 0 if self.mut_window is None else self.mut_window[0]
            end = L if self.mut_window is None else self.mut_window[1]
            window_size = end - start
            n_mutations = torch.floor(torch.tensor(window_size * self.mut_rate)).long()
            
            # Clone input for mutations
            batch_mut = batch_x.clone()
            
            # Track mutated positions
            mutated_positions = torch.zeros((batch_size, window_size), 
                                         dtype=torch.bool, device=self.device)
            
            # Apply mutations one at a time
            for _ in range(n_mutations):
                # Get attribution scores
                attr_map = self.attr_method(batch_mut)
                attr_map = attr_map[:, :, start:end]
                
                # Zero out current nucleotide scores and previously mutated positions
                current_nt_mask = batch_mut[:, :, start:end].bool()
                attr_map[current_nt_mask] = torch.tensor(float('-inf'))
                attr_map[:, :, mutated_positions] = torch.tensor(float('-inf'))
                
                # Apply temperature scaling if specified
                if self.temp > 0:
                    attr_map = attr_map / self.temp
                
                # Convert to probabilities and sample mutations
                probs = F.softmax(attr_map.reshape(batch_size, -1), dim=1)
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
                mutated_positions[batch_idx, mut_positions] = True
            
            # Move results back to CPU
            mutations.append(batch_mut.cpu())
            
        return torch.cat(mutations, dim=0)
    

class Sequential(Generator):
    """Generator that applies multiple generators in sequence.
    
    Args:
        generator_list (list[Generator]): List of generators to apply in sequence
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        
    Example:
        >>> g1 = Mutagenesis(mut_rate=0.1)
        >>> g2 = GuidedMutagenesis(attr_method, mut_rate=0.2)
        >>> seq_gen = Sequential([g1, g2])
        >>> mutated = seq_gen.generate(sequences)
    """
    def __init__(self, generator_list, seed=None):
        self.generator_list = generator_list
        if seed is not None:
            torch.manual_seed(seed)

    def generate(self, x):
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
        for generator in self.generator_list:
            x_mut = generator.generate(x_mut)
        return x_mut


class MultiGenerator(Generator):
    """Generator that applies multiple generators in parallel and combines results.
    
    Args:
        generator_list (list[Generator]): List of generators to apply in parallel
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        
    Example:
        >>> g1 = Mutagenesis(mut_rate=0.1)
        >>> g2 = GuidedMutagenesis(attr_method, mut_rate=0.2)
        >>> multi_gen = MultiGenerator([g1, g2])
        >>> mutated = multi_gen.generate(sequences)  # 2x batch size
    """
    def __init__(self, generator_list, seed=None):
        self.generator_list = generator_list
        if seed is not None:
            torch.manual_seed(seed)

    def generate(self, x):
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
        for generator in self.generator_list:
            outputs.append(generator.generate(x))
            
        # Combine results by concatenating along batch dimension
        x_mut = torch.cat(outputs, dim=0)
        
        return x_mut

