import torch

def upsample(x, min_needed=0, repeats=None):
    """Upsample sequences by repeating them to increase sample size.
    
    This function takes a batch of sequences and repeats them to create a larger dataset.
    The upsampling can be done either by specifying the minimum number of sequences needed
    or by directly setting the number of repeats per sequence.
    
    Parameters
    ----------
    x : torch.Tensor or array-like
        Input sequences of shape (N, A, L) where:
        N is batch size,
        A is alphabet size (e.g. 4 for DNA),
        L is sequence length
    min_needed : int, optional
        Minimum number of sequences needed in output. Used to calculate repeats if
        repeats parameter is None. By default 0
    repeats : int, optional
        Number of times to repeat each sequence. If None, calculated as ceiling division
        of min_needed by input batch size. By default None
        
    Returns
    -------
    torch.Tensor
        Upsampled sequences with shape (N * repeats, A, L). Each input sequence is
        repeated repeats times in the output
        
    Examples
    --------
    >>> x = torch.randn(10, 4, 200)  # 10 DNA sequences of length 200
    >>> upsampled = upsample(x, min_needed=100)  # Get at least 100 sequences
    >>> upsampled.shape
    torch.Size([100, 4, 200])
    """
    # Convert input to tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        
    # Calculate number of repeats needed if not specified
    if repeats is None:
        repeats = (min_needed + x.size(0) - 1) // x.size(0)  # Ceiling division
    
    # Use repeat_interleave for efficient upsampling
    return x.repeat_interleave(repeats, dim=0)


