"""Core functionality for tinybench."""

from tinybench.core.base import (
    BaseDataset,
    BaseMetric,
    BaseModel,
    BaseTransform,
)
from tinybench.core.benchmark import Benchmark
from tinybench.core.result import BenchmarkResult

__all__ = [
    "BaseModel",
    "BaseDataset",
    "BaseMetric",
    "BaseTransform",
    "Benchmark",
    "BenchmarkResult",
]

