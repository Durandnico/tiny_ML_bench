"""Base classes for mlbench library."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    """Abstract base class for all models.
    
    All model implementations should inherit from this class and implement
    the required methods: fit, predict, and optionally predict_proba for
    classification models.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model with hyperparameters.
        
        Args:
            **kwargs: Model-specific hyperparameters
        """
        self.params = kwargs
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> "BaseModel":
        """Fit the model to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        pass
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities (for classifiers).
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
            
        Raises:
            NotImplementedError: If the model doesn't support probability predictions
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability predictions"
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.params.copy()
    
    def set_params(self, **params: Any) -> "BaseModel":
        """Set model hyperparameters.
        
        Args:
            **params: Hyperparameters to set
            
        Returns:
            self: The model with updated parameters
        """
        self.params.update(params)
        return self
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class BaseDataset(ABC):
    """Abstract base class for datasets.
    
    All dataset implementations should inherit from this class and implement
    the required methods for loading and accessing data.
    """
    
    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the dataset.
        
        Args:
            name: Optional name for the dataset
        """
        self.name = name or self.__class__.__name__
        self.X_train: Optional[NDArray] = None
        self.X_test: Optional[NDArray] = None
        self.y_train: Optional[NDArray] = None
        self.y_test: Optional[NDArray] = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def load(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Load and return the dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        pass
    
    def get_train_data(self) -> Tuple[NDArray, NDArray]:
        """Get training data.
        
        Returns:
            Tuple of (X_train, y_train)
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[NDArray, NDArray]:
        """Get test data.
        
        Returns:
            Tuple of (X_test, y_test)
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.X_test, self.y_test
    
    def info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset metadata
        """
        info_dict = {
            "name": self.name,
            "n_train_samples": len(self.X_train) if self.X_train is not None else 0,
            "n_test_samples": len(self.X_test) if self.X_test is not None else 0,
            "n_features": self.X_train.shape[1] if self.X_train is not None else 0,
        }
        info_dict.update(self.metadata)
        return info_dict
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.
    
    All metric implementations should inherit from this class and implement
    the required compute method.
    """
    
    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the metric.
        
        Args:
            name: Optional name for the metric
        """
        self.name = name or self.__class__.__name__
        self.higher_is_better = True  # Override in subclasses if needed
    
    @abstractmethod
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Compute the metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or scores
            **kwargs: Additional metric-specific arguments
            
        Returns:
            Metric value (float) or dictionary of values
        """
        pass
    
    def __call__(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Compute the metric (callable interface).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or scores
            **kwargs: Additional metric-specific arguments
            
        Returns:
            Metric value(s)
        """
        return self.compute(y_true, y_pred, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseTransform(ABC):
    """Abstract base class for data transformations.
    
    All transformation implementations should inherit from this class.
    """
    
    def __init__(self) -> None:
        """Initialize the transform."""
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "BaseTransform":
        """Fit the transform to data.
        
        Args:
            X: Data to fit on
            y: Optional target values
            
        Returns:
            self: The fitted transform
        """
        pass
    
    @abstractmethod
    def transform(self, X: NDArray) -> NDArray:
        """Apply the transformation.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, X: NDArray, y: Optional[NDArray] = None) -> NDArray:
        """Fit and transform in one step.
        
        Args:
            X: Data to fit and transform
            y: Optional target values
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        return f"{self.__class__.__name__}()"
