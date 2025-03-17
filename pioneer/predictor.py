import torch

class Predictor:
    """Abstract base class for prediction methods.
    
    All predictor classes should inherit from this class and implement
    the predict method.
    """
    def __call__(self, model, x, batch_size=32):
        """Generate predictions for input sequences.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to use for predictions
        x : torch.Tensor
            Input sequences of shape (N, A, L)
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
    
    Parameters
    ----------
    task_index : int, optional
        Index to select from multi-task output.
        If None, assumes single task output, by default None
            
    Examples
    --------
    >>> predictor = Scalar(task_index=0)  # Select first task
    >>> scalar_preds = predictor.predict(model, sequences)
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
            Input sequences of shape (N, A, L)
        batch_size : int, optional
            Batch size for processing, by default 32
            
        Returns
        -------
        torch.Tensor
            Scalar predictions of shape (N,) or (N, T) for T tasks
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
    
    Parameters
    ----------
    reduction : callable, optional
        Function to reduce profiles to scalar values along the sequence length dimension.
        If None, defaults to torch.mean (averaging across sequence length), by default None
    task_index : int, optional
        Index to select from multi-task output.
        If None, assumes single task output, by default None
            
    Examples
    --------
    >>> predictor = Profile(task_index=1)  # Uses default mean reduction across sequence length
    >>> predictor = Profile(reduction=torch.sum, task_index=1)  # Sum across sequence length
    >>> scalar_preds = predictor.predict(model, sequences)
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
            Scalar predictions of shape (N,) or (N, T) where:
            N is batch size,
            T is number of tasks (if multi-task).
            The scalar values are obtained by applying the reduction function
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
    


