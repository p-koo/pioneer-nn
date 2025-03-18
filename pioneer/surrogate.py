import torch
from torch.utils.data import DataLoader, TensorDataset


class ModelWrapper(torch.nn.Module):
    """Model wrapper for predictions and uncertainty estimation.
    
    Parameters
    ----------
    model : torch.nn.Module or list[torch.nn.Module]
        Model or list of models for ensemble
    predictor : Predictor
        Prediction method for generating outputs
    uncertainty_method : UncertaintyMethod, optional
        Method for estimating prediction uncertainty, by default None
        
    Examples
    --------
    >>> model = ModelWrapper(
    ...     model=MyModel(),
    ...     predictor=ScalarPredictor(),
    ...     uncertainty_method=MCDropout()
    ... )
    """
    def __init__(self, model, predictor, uncertainty_method=None, batch_size:int=32):
        super().__init__()
        self.model = model
        self.predictor = predictor
        self.uncertainty_method = uncertainty_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size=batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, list):
            # Handle ensemble of models
            pred = torch.stack([
                self.predictor(m, x)
                for m in self.model
            ])
            # Average predictions across ensemble and move to CPU
            pred = pred.mean(dim=0)
        else:
            # Single model prediction
            pred = self.predictor(self.model, x)

        return(pred)

    def predict(self, x: torch.Tensor, auto_batch:bool=True) -> torch.Tensor:
        """Generate predictions using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
        
        auto_batch: bool
            If True, this method will automate batching. However this will tend to break gradients
            If False, this method will not autobatch but will retain gradients

        Returns
        -------
        torch.Tensor
            Model predictions of shape (N,) for single task
            or (N, T) for T tasks
        """
        
        if not auto_batch:
            # Move batch to GPU
            x = x.to(self.device)
            pred = self(x)

        else:
            x = TensorDataset(x)
            dl = DataLoader(x, batch_size=self.batch_size)
            pred = []
            with torch.no_grad():
                for batch in dl:
                    batch = batch[0]
                    batch = batch.to(self.device)
                    pred.append(self(batch).cpu())
                pred = torch.cat(pred)
                
        return pred

    def uncertainty(self, x: torch.Tensor, auto_batch:bool=True) -> torch.Tensor:
        """Generate uncertainty estimates using batched inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (N, A, L) where:
            N is batch size,
            A is alphabet size,
            L is sequence length
            
        Returns
        -------
        torch.Tensor
            Uncertainty scores of shape (N,)
            
        Raises
        ------
        ValueError
            If no uncertainty method was specified during initialization
        """
        if self.uncertainty_method is None:
            raise ValueError("No uncertainty method specified")
        
        if not auto_batch:
            # Move batch to GPU
            x = x.to(self.device)
            uncertainty = self.uncertainty_method(self.model, x)

        else:
            x = TensorDataset(x)
            dl = DataLoader(x, batch_size=self.batch_size)
            uncertainty = []
            with torch.no_grad():
                for batch in dl:
                    batch = batch[0]
                    batch = batch.to(self.device)
                    uncertainty.append(self.uncertainty_method(self.model, batch).cpu())
                uncertainty = torch.cat(uncertainty)

        return uncertainty


        