class Predictor:
    """Abstract base class for prediction methods.
    
    All predictor classes should inherit from this class and implement
    the predict method.
    """
    def predict(self, model, x, batch_size=32):
        """Generate predictions for input sequences.
        
        Args:
            model: PyTorch model to use for predictions
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing
            
        Returns:
            torch.Tensor: Model predictions
        """
        pass


class Scalar(Predictor):
    """Predictor for models that output scalar values directly.
    
    Args:
        task_index (int, optional): Index to select from multi-task output.
            If None, assumes single task output. Defaults to None.
            
    Example:
        >>> predictor = Scalar(task_index=0)  # Select first task
        >>> scalar_preds = predictor.predict(model, sequences)
    """
    def __init__(self, task_index=None):
        self.task_index = task_index

    def predict(self, model, x, batch_size=32):
        """Generate scalar predictions.
        
        Args:
            model: PyTorch model that outputs scalar values
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Scalar predictions of shape (N,) or (N, T) for T tasks
        """
        model.eval()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                pred = model(batch[0])
                # Handle multi-task output if task_index specified
                if self.task_index is not None:
                    pred = pred[:, self.task_index]
                predictions.append(pred)
                
        return torch.cat(predictions, dim=0)


class Profile(Predictor):
    """Predictor for models that output profiles, with reduction to scalar values.
    
    Args:
        reduction (callable): Function to reduce profiles to scalar values
        task_index (int, optional): Index to select from multi-task output.
            If None, assumes single task output. Defaults to None.
            
    Example:
        >>> predictor = Profile(reduction=profile_sum, task_index=1)
        >>> scalar_preds = predictor.predict(model, sequences)
    """
    def __init__(self, reduction, task_index=None):
        self.reduction = reduction
        self.task_index = task_index
        
    def predict(self, model, x, batch_size=32):
        """Generate predictions and reduce profiles to scalars.
        
        Args:
            model: PyTorch model that outputs profile predictions
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Scalar predictions of shape (N,) or (N, T) where:
                N is batch size
                T is number of tasks (if multi-task)
        """
        model.eval()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                # Model output shape: (N, T, L) where T is number of tasks
                pred = model(batch[0])
                
                # Handle multi-task output if task_index specified
                if self.task_index is not None:
                    # Select specific task: (N, L)
                    pred = pred[:, self.task_index]
                
                # Apply reduction to get scalar per sequence
                predictions.append(self.reduction(pred))
                
        return torch.cat(predictions, dim=0)
    