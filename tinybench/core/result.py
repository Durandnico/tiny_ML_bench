"""Result handling classes for benchmark experiments."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResult:
    """Container for a single benchmark experiment result.
    
    Attributes:
        model_name: Name of the model
        dataset_name: Name of the dataset
        metrics: Dictionary of metric names to scores
        train_time: Time taken to train the model (seconds)
        predict_time: Time taken to make predictions (seconds)
        run_id: Identifier for this run
        model_params: Dictionary of model hyperparameters
        metadata: Additional metadata
    """
    
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    train_time: float
    predict_time: float
    run_id: int = 0
    model_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "train_time": self.train_time,
            "predict_time": self.predict_time,
            "run_id": self.run_id,
            "model_params": self.model_params,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create result from dictionary.
        
        Args:
            data: Dictionary containing result data
            
        Returns:
            BenchmarkResult instance
        """
        return cls(**data)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"BenchmarkResult(\n"
            f"  Model: {self.model_name}\n"
            f"  Dataset: {self.dataset_name}\n"
            f"  Metrics: {metrics_str}\n"
            f"  Train time: {self.train_time:.4f}s\n"
            f"  Predict time: {self.predict_time:.4f}s\n"
            f")"
        )
