import torch

def upsample(x: torch.Tensor, min_needed: int = 0, repeats: Optional[int] = None) -> torch.Tensor:
    """Upsample sequences by repeating them.
    
    Args:
        x: Input tensor to upsample
        min_needed: Minimum number of sequences needed
        repeats: Number of times to repeat sequences. If None, calculated from min_needed.
        
    Returns:
        torch.Tensor of upsampled sequences
    """
    # Convert input to tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    # Calculate number of repeats needed
    if repeats is None:
        upsamples = ceil(min_needed / len(x))
    else:
        upsamples = repeats
    
    # Create indices for upsampling
    idx = torch.arange(len(x)).repeat(upsamples)
    
    # Return upsampled tensor
    return x[idx]