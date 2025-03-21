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
            A is alphabet size (e.g. 4 for DNA),
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
        If None, returns all task outputs.
        If provided, returns only the specified task's output.
        By default None
            
    Examples
    --------
    >>> # Single task prediction
    >>> predictor = Scalar()
    >>> preds = predictor(model, sequences)  # Shape: (N,)
    >>>
    >>> # Multi-task prediction, select first task
    >>> predictor = Scalar(task_index=0)
    >>> preds = predictor(model, sequences)  # Shape: (N,)
    >>>
    >>> # Multi-task prediction, return all tasks
    >>> predictor = Scalar()
    >>> preds = predictor(model, sequences)  # Shape: (N, T)
    """
    def __init__(self, task_index=None):
        self.task_index = task_index

    def __call__(self, model, x):
        """Generate scalar predictions.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model that outputs scalar values.
            For single task, output shape should be (N,).
            For multiple tasks, output shape should be (N, T).
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            For single task models:
                Shape (N,) containing scalar predictions
            For multi-task models with task_index=None:
                Shape (N, T) containing predictions for all T tasks
            For multi-task models with task_index specified:
                Shape (N,) containing predictions for selected task
                
        Raises
        ------
        AssertionError
            If task_index > 0 is specified but model returns single-task output
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
    and reduces them to scalar values using a specified reduction function. For multi-task models,
    it can select a specific task's output.

    Parameters
    ----------
    reduction : callable, optional
        Function to reduce profiles to scalar values along the sequence length dimension.
        Must accept a tensor and dim parameter.
        
        Common choices:
            - torch.mean: Average across sequence (default)
            - torch.sum: Sum across sequence
            - torch.max: Maximum value across sequence (returns values only)

        If None, defaults to torch.mean
    task_index : int, optional
        Index to select from multi-task output.
        If None, returns reduced values for all tasks.
        If provided, returns only the specified task's reduced values.
        By default None

    Examples
    --------
    >>> # Single task, mean reduction
    >>> predictor = Profile()
    >>> preds = predictor(model, sequences)  # Shape: (N,)
    >>>
    >>> # Multi-task, sum reduction, select second task
    >>> predictor = Profile(reduction=torch.sum, task_index=1)
    >>> preds = predictor(model, sequences)  # Shape: (N,)
    >>>
    >>> # Multi-task, max reduction, all tasks
    >>> predictor = Profile(reduction=lambda x,dim: torch.max(x,dim=dim)[0])
    >>> preds = predictor(model, sequences)  # Shape: (N, T)
    """
    def __init__(self, reduction=None, task_index=None):
        self.reduction = reduction if reduction is not None else torch.mean
        self.task_index = task_index
        
    def __call__(self, model, x):
        """Generate predictions and reduce profiles to scalars.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model that outputs profile predictions.
            For single task, output shape should be (N, L).
            For multiple tasks, output shape should be (N, T, L).
            Where L is sequence length and T is number of tasks.
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            For single task models:
                Shape (N,) containing reduced profile values
            For multi-task models with task_index=None:
                Shape (N, T) containing reduced values for all T tasks
            For multi-task models with task_index specified:
                Shape (N,) containing reduced values for selected task
                
        Raises
        ------
        AssertionError
            If task_index > 0 is specified but model returns single-task profile
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
