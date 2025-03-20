import torch

class Predictor:
    """Abstract base class for prediction methods.
    
    All predictor classes should inherit from this class and implement
    the __call__ method to generate predictions from a model.
    """
    def __call__(self, model, x, batch_size=32):
        """Generate predictions for input sequences.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to use for predictions
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing, by default 32
            
        Returns
        -------
        torch.Tensor
            Model predictions
        """
        pass


class Scalar(Predictor):
    """Predictor for models that output scalar values directly.
    
    This predictor handles models that output scalar values for each input sequence.
    For multi-task models, it can select a specific task's output.
    
    Parameters
    ----------
    task_index : int, optional
        Index to select from multi-task output.
        If None, assumes single task output, by default None
            
    Examples
    --------
    >>> predictor = Scalar(task_index=0)  # Select first task
    >>> scalar_preds = predictor(model, sequences)  # Get predictions
    """
    def __init__(self, task_index=None):
        self.task_index = task_index

    def __call__(self, model, x):
        """Generate scalar predictions.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model that outputs scalar values
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Scalar predictions of shape (N,) for single task
            or (N, T) for T tasks if task_index is None
            or (N,) for specific task if task_index is provided
        """
       
        pred = model(x)
        # Handle multi-task output if task_index specified
        if self.task_index is not None and pred.ndim == 2:
            pred = pred[:, self.task_index]
        elif self.task_index is not None and pred.ndim == 1:
            assert self.task_index == 0, 'Task index is >0 but model does not return a multi-task prediction'
        
        return pred


class Profile(Predictor):
    """Predictor for models that output profiles, with reduction to scalar values.
    
    This predictor handles models that output profile predictions (values across sequence positions)
    and reduces them to scalar values. For multi-task models, it can select a specific task's output.
    
    Parameters
    ----------
    reduction : callable, optional
        Function to reduce profiles to scalar values along the sequence length dimension.
        Common choices include torch.mean, torch.sum, torch.max.
        If None, defaults to torch.mean (averaging across sequence length), by default None
    task_index : int, optional
        Index to select from multi-task output.
        If None, assumes single task output, by default None
            
    Examples
    --------
    >>> predictor = Profile(task_index=1)  # Uses default mean reduction across sequence length
    >>> predictor = Profile(reduction=torch.sum, task_index=1)  # Sum across sequence length
    >>> scalar_preds = predictor(model, sequences)  # Get predictions
    """
    def __init__(self, reduction=None, task_index=None):
        self.reduction = reduction if reduction is not None else torch.mean
        self.task_index = task_index
        
    def __call__(self, model, x):
        """Generate predictions and reduce profiles to scalars.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model that outputs profile predictions
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Scalar predictions of shape (N,) for single task
            or (N, T) for T tasks if task_index is None
            or (N,) for specific task if task_index is provided.
            Values are obtained by applying the reduction function
            across the sequence length dimension.
        """
        # Model output shape: (N, T, L) or (N, L) where:
        # N is batch size, T is number of tasks, L is sequence length
        pred = model(x)
            
        # Handle multi-task output if task_index specified
        if self.task_index is not None and pred.ndim == 3:
            # Select specific task: (N, L) from (N, T, L)
            pred = pred[:, self.task_index, :]
        elif self.task_index is not None and pred.ndim == 2:
            assert self.task_index == 0, 'Task index is >0 but model does not return a multi-task profile'
        # Apply reduction across sequence length dimension to get scalar per sequence
        pred = self.reduction(pred, dim=-1)

        return pred
