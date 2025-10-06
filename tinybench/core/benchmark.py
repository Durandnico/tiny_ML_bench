"""Benchmark orchestration for model evaluation."""
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from tinybench.core.base import BaseDataset, BaseMetric, BaseModel
from tinybench.core.result import BenchmarkResult


class Benchmark:
    """Orchestrates benchmarking of models on datasets with metrics.
    
    This class handles the execution of benchmarking experiments, including
    training models, making predictions, computing metrics, and tracking timing.
    """
    
    def __init__(
        self,
        models: Union[BaseModel, List[BaseModel]],
        datasets: Union[BaseDataset, List[BaseDataset]],
        metrics: Union[BaseMetric, List[BaseMetric]],
        n_runs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the benchmark.
        
        Args:
            models: Model(s) to benchmark
            datasets: Dataset(s) to use for benchmarking
            metrics: Metric(s) to compute
            n_runs: Number of times to run each experiment
            random_state: Random seed for reproducibility
        """
        self.models = [models] if isinstance(models, BaseModel) else models
        self.datasets = [datasets] if isinstance(datasets, BaseDataset) else datasets
        self.metrics = [metrics] if isinstance(metrics, BaseMetric) else metrics
        self.n_runs = n_runs
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.results: List[BenchmarkResult] = []
    
    def run(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Run the benchmark experiments.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            List of BenchmarkResult objects
        """
        self.results = []
        
        for dataset in self.datasets:
            if verbose:
                print(f"\nBenchmarking on dataset: {dataset.name}")
            
            # Load dataset
            X_train, X_test, y_train, y_test = dataset.load()
            
            for model in self.models:
                if verbose:
                    print(f"  Model: {model.__class__.__name__}")
                
                for run in range(self.n_runs):
                    if verbose and self.n_runs > 1:
                        print(f"    Run {run + 1}/{self.n_runs}")
                    
                    result = self._run_single_experiment(
                        model=model,
                        dataset=dataset,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        run_id=run,
                    )
                    
                    self.results.append(result)
                    
                    if verbose:
                        self._print_result(result)
        
        return self.results
    
    def _run_single_experiment(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        X_train: NDArray,
        X_test: NDArray,
        y_train: NDArray,
        y_test: NDArray,
        run_id: int,
    ) -> BenchmarkResult:
        """Run a single benchmark experiment.
        
        Args:
            model: Model to benchmark
            dataset: Dataset being used
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            run_id: Identifier for this run
            
        Returns:
            BenchmarkResult containing experiment results
        """
        # Train model
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start
        
        # Make predictions
        predict_start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - predict_start
        
        # Compute metrics
        metric_scores: Dict[str, float] = {}
        for metric in self.metrics:
            try:
                score = metric.compute(y_test, y_pred)
                if isinstance(score, dict):
                    metric_scores.update(score)
                else:
                    metric_scores[metric.name] = score
            except Exception as e:
                print(f"    Warning: Failed to compute {metric.name}: {e}")
                metric_scores[metric.name] = np.nan
        
        # Create result object
        result = BenchmarkResult(
            model_name=model.__class__.__name__,
            dataset_name=dataset.name,
            metrics=metric_scores,
            train_time=train_time,
            predict_time=predict_time,
            run_id=run_id,
            model_params=model.get_params(),
        )
        
        return result
    
    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a single result.
        
        Args:
            result: BenchmarkResult to print
        """
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in result.metrics.items()
        )
        print(f"      {metrics_str}")
        print(f"      Train time: {result.train_time:.4f}s, "
              f"Predict time: {result.predict_time:.4f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark results.
        
        Returns:
            Dictionary containing aggregated results
        """
        if not self.results:
            return {"error": "No results available. Run benchmark first."}
        
        summary: Dict[str, Any] = {}
        
        # Group results by model and dataset
        for result in self.results:
            key = f"{result.model_name}_{result.dataset_name}"
            
            if key not in summary:
                summary[key] = {
                    "model": result.model_name,
                    "dataset": result.dataset_name,
                    "metrics": {m: [] for m in result.metrics.keys()},
                    "train_times": [],
                    "predict_times": [],
                }
            
            for metric_name, value in result.metrics.items():
                summary[key]["metrics"][metric_name].append(value)
            summary[key]["train_times"].append(result.train_time)
            summary[key]["predict_times"].append(result.predict_time)
        
        # Compute aggregated statistics
        for key in summary:
            for metric_name in summary[key]["metrics"]:
                values = summary[key]["metrics"][metric_name]
                summary[key]["metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            
            summary[key]["train_time_mean"] = np.mean(summary[key]["train_times"])
            summary[key]["predict_time_mean"] = np.mean(summary[key]["predict_times"])
            del summary[key]["train_times"]
            del summary[key]["predict_times"]
        
        return summary
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to a file.
        
        Args:
            filepath: Path where to save results
        """
        import json
        
        results_data = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def __repr__(self) -> str:
        """String representation of the benchmark."""
        return (
            f"Benchmark(n_models={len(self.models)}, "
            f"n_datasets={len(self.datasets)}, "
            f"n_metrics={len(self.metrics)}, "
            f"n_runs={self.n_runs})"
        )
