import torch
from torch.utils.data import DataLoader, TensorDataset


class Oracle:
    """Abstract base class for experimental oracles.
    
    All oracle classes should inherit from this class and implement
    the predict method to generate ground truth labels for input sequences.
    """
    def predict(self, x, batch_size=32):
        """Generate ground truth labels for input sequences.
        
        Parameters
        ----------
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
            Ground truth labels of shape (N,1) for single task
            or (N,T) for T tasks
        """
        raise NotImplementedError


class SingleOracle(Oracle):
    """Oracle that uses a single pre-trained model for inference.
    
    This class wraps a single pre-trained model to act as an oracle for generating
    predictions on input sequences. It handles model loading, device placement,
    and batched inference.
    
    Parameters
    ----------
    model_class : type
        PyTorch model class to instantiate
    model_kwargs : dict
        Arguments to initialize the model
    weight_path : str
        Path to model weights file
    predictor : pioneer.predictor.Predictor
        Prediction method for generating outputs
    device : str, optional
        Device to run model on ('cuda' or 'cpu').
        Defaults to 'cuda' if available
    model_type : str, optional
        Type of model to use ('pytorch' or 'lightning').
        If 'lightning', loads from checkpoint file.
        If 'pytorch', loads from state_dict file.
        By default 'pytorch'
            
    Examples
    --------
    >>> model_class = MyModel
    >>> model_kwargs = {'hidden_dim': 256}
    >>> predictor = Scalar()  # For scalar outputs
    >>> oracle = SingleOracle(model_class, model_kwargs, 'weights.pt', predictor)
    >>> labels = oracle.predict(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_path, predictor, device=None, model_type='pytorch'):
        # Set device (default to GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load model
        if model_type == 'pytorch':
            self.model = model_class(**model_kwargs)
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        elif model_type == 'lightning':
            self.model = model_class.load_from_checkpoint(weight_path, **model_kwargs)
            self.model.to(self.device)
        else:
            raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
        self.model.eval()
        self.predictor = predictor

    def predict(self, x, batch_size=32):
        """Generate predictions using batched inference.
        
        This method processes input sequences in batches to generate predictions
        efficiently while managing memory usage.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease if running into 
            GPU memory issues. By default 32
                
        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,1) for single task
            or (N,T) for T tasks
        """
        predictions = []
        
        # Create DataLoader for batched processing
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in loader:
                # Move batch to GPU, get predictions, move back to CPU
                batch_x = batch_x.to(self.device)
                pred = self.predictor(self.model, batch_x).cpu()
                predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)
        if predictions.ndim == 1:
            return predictions.unsqueeze(-1)
        else:
            return predictions


class EnsembleOracle(Oracle):
    """Oracle that uses an ensemble of pre-trained models.
    
    This class manages multiple pre-trained models as an ensemble to generate
    predictions and uncertainty estimates. It handles model loading, device
    placement, and batched inference across the ensemble.
    
    Parameters
    ----------
    model_class : type
        PyTorch model class to instantiate
    model_kwargs : dict
        Arguments to initialize each model
    weight_paths : list[str]
        Paths to model weight files
    predictor : pioneer.predictor.Predictor
        Prediction method for generating outputs
    device : str, optional
        Device to run models on ('cuda' or 'cpu').
        Defaults to 'cuda' if available
    model_type : str, optional
        Type of model to use ('pytorch' or 'lightning').
        If 'lightning', loads from checkpoint files.
        If 'pytorch', loads from state_dict files.
        By default 'pytorch'
            
    Examples
    --------
    >>> model_class = MyModel
    >>> model_kwargs = {'hidden_dim': 256}
    >>> weight_paths = ['model1.pt', 'model2.pt', 'model3.pt']
    >>> predictor = Profile(reduction=torch.mean)  # For profile outputs
    >>> oracle = EnsembleOracle(model_class, model_kwargs, weight_paths, predictor)
    >>> labels = oracle.predict(sequences)
    >>> labels, uncertainties = oracle.predict_uncertainty(sequences)
    """
    def __init__(self, model_class, model_kwargs, weight_paths, predictor, device=None, model_type='pytorch'):
        # Set device (default to GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load all models
        models = []
        for path in weight_paths:
            
            if model_type == 'pytorch':
                model = model_class(**model_kwargs)
                model.to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
            elif model_type == 'lightning':
                model = model_class.load_from_checkpoint(path, **model_kwargs)
                model.to(self.device)
            else:
                raise ValueError(f"Invalid model type: {model_type}, must be 'pytorch' or 'lightning'")
            model.eval()
            models.append(model)
        self.models = torch.nn.ModuleList(models)
        self.predictor = predictor

    def full_predict(self, x, batch_size=32):
        """Generate raw ensemble predictions using batched inference.
        
        This method returns predictions from each model in the ensemble
        before averaging, useful for uncertainty estimation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease if running into 
            GPU memory issues. By default 32
                
        Returns
        -------
        torch.Tensor
            Raw predictions from each model of shape (E, N, T) where:
            E is ensemble size,
            N is batch size,
            T is number of tasks (or 1 for single task)
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
                    self.predictor(model, batch_x)
                    for model in self.models
                ])
                
                # Average predictions across ensemble and move to CPU
                pred = batch_preds.cpu()
                predictions.append(pred)
                
        return torch.cat(predictions, dim=1)
    
    def predict(self, x, batch_size=32):
        """Generate ensemble predictions using batched inference.
        
        This method returns the mean predictions across the ensemble.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease if running into 
            GPU memory issues. By default 32
                
        Returns
        -------
        torch.Tensor
            Mean predictions across ensemble of shape (N,1) for single task
            or (N,T) for T tasks
        """
        predictions = self.full_predict(x, batch_size).mean(dim=0)
        if predictions.ndim == 1:
            return predictions.unsqueeze(-1)
        else:
            return predictions
    
    
    def predict_uncertainty(self, x, batch_size=32):
        """Generate ensemble predictions and uncertainty using batched inference.
        
        This method returns both the mean predictions and standard deviations
        across the ensemble, providing uncertainty estimates.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size (e.g. 4 for DNA),
            L is sequence length
        batch_size : int, optional
            Batch size for processing. Decrease if running into 
            GPU memory issues. By default 32
                
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Mean predictions across ensemble of shape (N,1) for single task
              or (N,T) for T tasks
            - Standard deviations across ensemble of shape (N,1) for single task
              or (N,T) for T tasks, representing prediction uncertainty
        """
        predictions = self.full_predict(x, batch_size)
        means = predictions.mean(dim=0)
        stds = predictions.std(dim=0)
        if means.ndim == 1:
            means = means.unsqueeze(-1)
        if stds.ndim == 1:
            stds = stds.unsqueeze(-1)
        
        return means, stds

