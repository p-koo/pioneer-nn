import torch

def upsample(x, min_needed=0, repeats=None):
    """Upsample sequences by repeating them to increase sample size.
    
    Parameters
    ----------
    x : torch.Tensor or array-like
        Input sequences of shape (N, A, L) where:
        N is batch size,
        A is alphabet size,
        L is sequence length
    min_needed : int, optional
        Minimum number of sequences needed, by default 0
    repeats : int, optional
        Number of times to repeat sequences. If None, calculated from min_needed,
        by default None
        
    Returns
    -------
    torch.Tensor
        Upsampled sequences with shape (N * repeats, A, L)
    """
    # Convert input to tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        
    # Calculate number of repeats needed if not specified
    if repeats is None:
        repeats = (min_needed + x.size(0) - 1) // x.size(0)  # Ceiling division
    
    # Use repeat_interleave for efficient upsampling
    return x.repeat_interleave(repeats, dim=0)


