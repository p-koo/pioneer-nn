import torch
from torch.utils.data import DataLoader, TensorDataset


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


class ScalarPredictor(Predictor):
    """Predictor for models that output scalar values directly.
    
    Example:
        >>> predictor = ScalarPredictor()
        >>> scalar_preds = predictor.predict(model, sequences)
    """
    def predict(self, model, x, batch_size=32):
        """Generate scalar predictions.
        
        Args:
            model: PyTorch model that outputs scalar values
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Scalar predictions of shape (N,)
        """
        model.eval()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                predictions.append(model(batch[0]))
                
        return torch.cat(predictions, dim=0)


class ProfilePredictor(Predictor):
    """Predictor for models that output profiles, with reduction to scalar values.
    
    Args:
        reduction (callable): Function to reduce profiles to scalar values
        
    Example:
        >>> predictor = ProfilePredictor(reduction=profile_sum)
        >>> scalar_preds = predictor.predict(model, sequences)
    """
    def __init__(self, reduction):
        self.reduction = reduction
        
    def predict(self, model, x, batch_size=32):
        """Generate predictions and reduce profiles to scalars.
        
        Args:
            model: PyTorch model that outputs profile predictions
            x (torch.Tensor): Input sequences of shape (N, A, L)
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            torch.Tensor: Scalar predictions of shape (N,)
        """
        model.eval()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                pred = model(batch[0])
                predictions.append(self.reduction(pred))
                
        return torch.cat(predictions, dim=0)


def profile_sum(pred):
    """Reduce profile predictions to scalars using summation.
    
    Args:
        pred (torch.Tensor): Profile predictions of shape (N, P) where:
            N is batch size
            P is profile length
            
    Returns:
        torch.Tensor: Summed predictions of shape (N,)
    """
    return torch.sum(pred, dim=1)


def profile_pca(pred):
    """Reduce profile predictions to scalars using PCA.
    
    Args:
        pred (torch.Tensor): Profile predictions of shape (N, P) where:
            N is batch size
            P is profile length
            
    Returns:
        torch.Tensor: First principal component of shape (N,)
    """
    # Reshape if needed
    if pred.dim() > 2:
        pred = pred.reshape(pred.shape[0], -1)
    
    # Center the data
    mean = torch.mean(pred, dim=0)
    centered = pred - mean
    
    # Compute first principal component
    U, S, V = torch.svd(centered.T)
    projection = centered @ U[:, 0]
    
    # Correct sign based on correlation with sum
    sums = torch.sum(pred, dim=1)
    if torch.corrcoef(torch.stack([sums, projection]))[0, 1] < 0:
        projection = -projection
        
    return projection

