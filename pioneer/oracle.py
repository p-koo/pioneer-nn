import torch
from torch.utils.data import DataLoader, TensorDataset


class Oracle:
    """Abstract base class for experimental oracles.
    
    All oracle classes should inherit from this class and implement
    the predict method.
    """
    def predict(self, x, batch_size=32):
        """Generate ground truth labels for input sequences.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Ground truth labels of shape (N,)
        """
        raise NotImplementedError


class SingleOracle(Oracle):
    """Oracle that uses a single pre-trained model for inference.
    
    Args:
        model_class: PyTorch model class to instantiate
        model_kwargs (dict): Arguments to initialize the model
        weight_path (str): Path to model weights file
        device (str, optional): Device to run model on ('cuda' or 'cpu'). 
            Defaults to 'cuda' if available.
            
    Example:
        >>> model_class = MyModel
        >>> model_kwargs = {'hidden_dim': 256}
        >>> oracle = SingleOracle(model_class, model_kwargs, 'weights.pt')
        >>> labels = oracle.predict(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_path, device=None):
        # Set device (default to GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load model
        self.model = model_class(**model_kwargs)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def predict(self, x, batch_size=32):
        """Generate predictions using batched inference.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Model predictions of shape (N,)
        """
        # Process sequences in batches
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                # Generate predictions for current batch
                pred = self.model(batch[0])
                predictions.append(pred)
                
        # Combine all batch results
        return torch.cat(predictions, dim=0)


class EnsembleOracle(Oracle):
    """Oracle that uses an ensemble of pre-trained models.
    
    Args:
        model_class: PyTorch model class to instantiate
        model_kwargs (dict): Arguments to initialize each model
        weight_paths (list[str]): Paths to model weight files
        device (str, optional): Device to run models on ('cuda' or 'cpu').
            Defaults to 'cuda' if available.
            
    Example:
        >>> model_class = MyModel
        >>> model_kwargs = {'hidden_dim': 256}
        >>> weight_paths = ['model1.pt', 'model2.pt', 'model3.pt']
        >>> oracle = EnsembleOracle(model_class, model_kwargs, weight_paths)
        >>> labels = oracle.predict(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_paths, device=None):
        # Set device (default to GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load all models
        self.models = []
        for path in weight_paths:
            model = model_class(**model_kwargs)
            model.to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.models.append(model)

    def predict(self, x, batch_size=32):
        """Generate ensemble predictions using batched inference.
        
        Args:
            x (torch.Tensor): Input sequences of shape (N, A, L) where:
                N is batch size
                A is alphabet size
                L is sequence length
            batch_size (int, optional): Batch size for processing. Defaults to 32.
                
        Returns:
            torch.Tensor: Mean predictions across ensemble of shape (N,)
        """
        # Process sequences in batches
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                # Get predictions from all models
                preds = torch.stack([
                    model(batch[0]) for model in self.models
                ])
                # Average predictions across ensemble
                pred = torch.mean(preds, dim=0)
                predictions.append(pred)
                
        # Combine all batch results
        return torch.cat(predictions, dim=0)