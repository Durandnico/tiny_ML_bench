"""Classification metrics."""

from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tinybench.core.base import BaseMetric


class Accuracy(BaseMetric):
    """Accuracy classification metric."""
    
    def __init__(self) -> None:
        """Initialize accuracy metric."""
        super().__init__(name="accuracy")
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            **kwargs: Additional arguments
            
        Returns:
            Accuracy score
        """
        return float(accuracy_score(y_true, y_pred))


class Precision(BaseMetric):
    """Precision classification metric."""
    
    def __init__(self, average: str = "binary") -> None:
        """Initialize precision metric.
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        super().__init__(name="precision")
        self.average = average
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute precision score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            **kwargs: Additional arguments
            
        Returns:
            Precision score
        """
        return float(precision_score(y_true, y_pred, average=self.average, zero_division=0))


class Recall(BaseMetric):
    """Recall classification metric."""
    
    def __init__(self, average: str = "binary") -> None:
        """Initialize recall metric.
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        super().__init__(name="recall")
        self.average = average
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            **kwargs: Additional arguments
            
        Returns:
            Recall score
        """
        return float(recall_score(y_true, y_pred, average=self.average, zero_division=0))


class F1Score(BaseMetric):
    """F1 score classification metric."""
    
    def __init__(self, average: str = "binary") -> None:
        """Initialize F1 score metric.
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        super().__init__(name="f1_score")
        self.average = average
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            **kwargs: Additional arguments
            
        Returns:
            F1 score
        """
        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))


class ROC_AUC(BaseMetric):
    """ROC AUC classification metric."""
    
    def __init__(self, average: str = "macro") -> None:
        """Initialize ROC AUC metric.
        
        Args:
            average: Averaging strategy for multi-class
        """
        super().__init__(name="roc_auc")
        self.average = average
        self.higher_is_better = True
    
    def compute(
        self, 
        y_true: NDArray, 
        y_pred: NDArray,
        **kwargs: Any
    ) -> float:
        """Compute ROC AUC score.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities or scores
            **kwargs: Additional arguments
            
        Returns:
            ROC AUC score
        """
        try:
            return float(roc_auc_score(y_true, y_pred, average=self.average, multi_class="ovr"))
        except ValueError:
            return np.nan


