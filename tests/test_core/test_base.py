"""Tests for base classes."""

import pytest
import numpy as np

from tinybench.core.base import BaseModel, BaseDataset, BaseMetric, BaseTransform


class TestBaseModel:
    """Tests for BaseModel class."""
    
    def test_base_model_cannot_instantiate(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_model_get_set_params(self, simple_classification_model):
        """Test getting and setting model parameters."""
        model = simple_classification_model
        
        # Test get_params
        params = model.get_params()
        assert isinstance(params, dict)
        
        # Test set_params
        model.set_params(test_param=123)
        assert model.params.get("test_param") == 123
    
    def test_model_fit_predict(self, simple_classification_model, classification_data):
        """Test fitting and predicting with a model."""
        model = simple_classification_model
        X_train, X_test, y_train, y_test = classification_data
        
        # Test fit
        model.fit(X_train, y_train)
        assert model.is_fitted
        
        # Test predict
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert y_pred.shape == y_test.shape
    
    def test_model_repr(self, simple_classification_model):
        """Test model string representation."""
        model = simple_classification_model
        repr_str = repr(model)
        assert "SimpleClassifier" in repr_str


class TestBaseDataset:
    """Tests for BaseDataset class."""
    
    def test_base_dataset_cannot_instantiate(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()
    
    def test_dataset_load(self, simple_dataset):
        """Test loading a dataset."""
        dataset = simple_dataset
        X_train, X_test, y_train, y_test = dataset.load()
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_dataset_get_train_data(self, simple_dataset):
        """Test getting training data."""
        dataset = simple_dataset
        dataset.load()
        
        X_train, y_train = dataset.get_train_data()
        assert X_train is not None
        assert y_train is not None
        assert len(X_train) == len(y_train)
    
    def test_dataset_get_test_data(self, simple_dataset):
        """Test getting test data."""
        dataset = simple_dataset
        dataset.load()
        
        X_test, y_test = dataset.get_test_data()
        assert X_test is not None
        assert y_test is not None
        assert len(X_test) == len(y_test)
    
    def test_dataset_info(self, simple_dataset):
        """Test getting dataset info."""
        dataset = simple_dataset
        dataset.load()
        
        info = dataset.info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "n_train_samples" in info
        assert "n_test_samples" in info
        assert "n_features" in info
        
        assert info["n_train_samples"] > 0
        assert info["n_test_samples"] > 0
        assert info["n_features"] > 0
    
    def test_dataset_repr(self, simple_dataset):
        """Test dataset string representation."""
        dataset = simple_dataset
        repr_str = repr(dataset)
        assert "SimpleDataset" in repr_str


class TestBaseMetric:
    """Tests for BaseMetric class."""
    
    def test_base_metric_cannot_instantiate(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()
    
    def test_metric_compute(self, simple_metric, classification_data):
        """Test computing a metric."""
        metric = simple_metric
        _, _, y_train, y_test = classification_data
        
        # Use same values for perfect score
        score = metric.compute(y_train, y_train)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
    
    def test_metric_callable(self, simple_metric, classification_data):
        """Test that metric is callable."""
        metric = simple_metric
        _, _, y_train, _ = classification_data
        
        # Test __call__ method
        score = metric(y_train, y_train)
        assert isinstance(score, (int, float))
    
    def test_metric_repr(self, simple_metric):
        """Test metric string representation."""
        metric = simple_metric
        repr_str = repr(metric)
        assert "SimpleMetric" in repr_str or "accuracy" in repr_str.lower()


class TestBaseTransform:
    """Tests for BaseTransform class."""
    
    def test_base_transform_cannot_instantiate(self):
        """Test that BaseTransform cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTransform()
    
    def test_transform_fit_transform(self, classification_data):
        """Test fitting and transforming data."""
        from sklearn.preprocessing import StandardScaler
        
        class SimpleTransform(BaseTransform):
            def __init__(self):
                super().__init__()
                self.scaler = StandardScaler()
            
            def fit(self, X, y=None):
                self.scaler.fit(X)
                self.is_fitted = True
                return self
            
            def transform(self, X):
                return self.scaler.transform(X)
        
        X_train, X_test, _, _ = classification_data
        transform = SimpleTransform()
        
        # Test fit
        transform.fit(X_train)
        assert transform.is_fitted
        
        # Test transform
        X_transformed = transform.transform(X_test)
        assert X_transformed.shape == X_test.shape
        
        # Test fit_transform
        transform2 = SimpleTransform()
        X_fit_transformed = transform2.fit_transform(X_train)
        assert X_fit_transformed.shape == X_train.shape
        assert transform2.is_fitted
