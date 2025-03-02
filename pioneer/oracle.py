import torch
from torch.utils.data import DataLoader, TensorDataset


class Oracle:
    """Abstract base class for experimental oracles.
    
    All oracle classes should inherit from this class and implement
    the predict method.
    """
    def predict(self, x, batch_size=32):
        """Generate ground truth labels for input sequences.
        
        Parameters
        ----------
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
            Ground truth labels of shape (N,) for single task
            or (N, T) for T tasks
        """
        raise NotImplementedError


class SingleOracle(Oracle):
    """Oracle that uses a single pre-trained model for inference.
    
    Parameters
    ----------
    model_class : type
        PyTorch model class to instantiate
    model_kwargs : dict
        Arguments to initialize the model
    weight_path : str
        Path to model weights file
    predictor : Predictor
        Prediction method for generating outputs
    device : str, optional
        Device to run model on ('cuda' or 'cpu').
        Defaults to 'cuda' if available, by default None
            
    Examples
    --------
    >>> model_class = MyModel
    >>> model_kwargs = {'hidden_dim': 256}
    >>> predictor = Scalar()  # For scalar outputs
    >>> oracle = SingleOracle(model_class, model_kwargs, 'weights.pt', predictor)
    >>> labels = oracle.predict(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_path, predictor, device=None):
        # Set device (default to GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load model
        self.model = model_class(**model_kwargs)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        self.predictor = predictor

    def predict(self, x, batch_size=32):
        """Generate predictions using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
                
        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,) for single task
            or (N, T) for T tasks
        """
        predictions = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU, get predictions, move back to CPU
                batch_x = batch_x.to(self.device)
                pred = self.predictor.predict(self.model, batch_x).cpu()
                predictions.append(pred)
                
        return torch.cat(predictions, dim=0)


class EnsembleOracle(Oracle):
    """Oracle that uses an ensemble of pre-trained models.
    
    Parameters
    ----------
    model_class : type
        PyTorch model class to instantiate
    model_kwargs : dict
        Arguments to initialize each model
    weight_paths : list[str]
        Paths to model weight files
    predictor : Predictor
        Prediction method for generating outputs
    device : str, optional
        Device to run models on ('cuda' or 'cpu').
        Defaults to 'cuda' if available, by default None
            
    Examples
    --------
    >>> model_class = MyModel
    >>> model_kwargs = {'hidden_dim': 256}
    >>> weight_paths = ['model1.pt', 'model2.pt', 'model3.pt']
    >>> predictor = Profile(reduction=torch.mean)  # For profile outputs
    >>> oracle = EnsembleOracle(model_class, model_kwargs, weight_paths, predictor)
    >>> labels = oracle.predict(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_paths, predictor, device=None):
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
        self.predictor = predictor

    def predict(self, x, batch_size=32):
        """Generate ensemble predictions using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease this value if running into 
            GPU memory issues, by default 32
                
        Returns
        -------
        torch.Tensor
            Mean predictions across ensemble of shape (N,) for single task
            or (N, T) for T tasks
        """
        predictions = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU
                batch_x = batch_x.to(self.device)
                
                # Get predictions from all models
                batch_preds = torch.stack([
                    self.predictor.predict(model, batch_x)
                    for model in self.models
                ])
                
                # Average predictions across ensemble and move to CPU
                pred = batch_preds.mean(dim=0).cpu()
                predictions.append(pred)
                
        return torch.cat(predictions, dim=0)


    