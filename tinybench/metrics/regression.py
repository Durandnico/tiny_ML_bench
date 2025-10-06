"""Regression metrics."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tinybench.core.base import BaseMetric


class MeanSquaredError(BaseMetric):
    """Mean Squared Error regression metric."""
    
    def __init__(self) -> None:
        """Initialize MSE metric."""
        super().__init__(name="mse")
        self.higher_is_better = False
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute MSE.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments
            
        Returns:
            MSE value
        """
        return float(mean_squared_error(y_true, y_pred))


class RootMeanSquaredError(BaseMetric):
    """Root Mean Squared Error regression metric."""
    
    def __init__(self) -> None:
        """Initialize RMSE metric."""
        super().__init__(name="rmse")
        self.higher_is_better = False
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute RMSE.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class MeanAbsoluteError(BaseMetric):
    """Mean Absolute Error regression metric."""
    
    def __init__(self) -> None:
        """Initialize MAE metric."""
        super().__init__(name="mae")
        self.higher_is_better = False
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute MAE.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments
            
        Returns:
            MAE value
        """
        return float(mean_absolute_error(y_true, y_pred))


class R2Score(BaseMetric):
    """R-squared (coefficient of determination) metric."""
    
    def __init__(self) -> None:
        """Initialize R² metric."""
        super().__init__(name="r2")
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute R² score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments
            
        Returns:
            R² score
        """
        return float(r2_score(y_true, y_pred))
