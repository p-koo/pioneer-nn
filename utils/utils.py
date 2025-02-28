class UpSelector(Selector):
    """Selector that upsamples sequences by repeating them.
    
    Attributes:
        min_needed: Minimum number of sequences needed
        repeats: Number of times to repeat sequences. If None, calculated from min_needed.
    """
    def __init__(self, min_needed:int=0, repeats:Optional[int]=None) -> None:
        super().__init__(min_needed, None)
        self.repeats = repeats

    def make_idx(self, x:Union[np.ndarray,torch.Tensor]) -> np.ndarray:
        """Generate indices that repeat the input sequences.
        
        Args:
            x: Input array to select from
            
        Returns:
            numpy.ndarray of indices that repeat the sequences
        """
        if self.repeats is None:
            upsamples = ceil(self.n_to_get/len(x))
        else:
            upsamples = self.repeats
        
        idx = np.concatenate([np.arange(len(x)) for i in range(upsamples)])
        return idx