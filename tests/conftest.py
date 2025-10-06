"""Pytest configuration and fixtures for tinybench tests."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tinybench.core.base import BaseDataset, BaseMetric, BaseModel


@pytest.fixture
def random_seed():
    """Fixture for random seed."""
    return 42


@pytest.fixture
def classification_data(random_seed):
    """Fixture for classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=random_seed,
    )
    
    # Split into train/test
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data(random_seed):
    """Fixture for regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=random_seed,
    )
    
    # Split into train/test
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def simple_classification_model():
    """Fixture for a simple classification model."""
    from sklearn.linear_model import LogisticRegression
    
    class SimpleClassifier(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = LogisticRegression(random_state=42)
        
        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted = True
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    
    return SimpleClassifier()


@pytest.fixture
def simple_regression_model():
    """Fixture for a simple regression model."""
    from sklearn.linear_model import LinearRegression
    
    class SimpleRegressor(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = LinearRegression()
        
        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted = True
            return self
        
        def predict(self, X):
            return self.model.predict(X)
    
    return SimpleRegressor()


@pytest.fixture
def simple_dataset(classification_data):
    """Fixture for a simple dataset."""
    X_train, X_test, y_train, y_test = classification_data
    
    class SimpleDataset(BaseDataset):
        def __init__(self):
            super().__init__(name="SimpleDataset")
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        
        def load(self):
            return self.X_train, self.X_test, self.y_train, self.y_test
    
    return SimpleDataset()


@pytest.fixture
def simple_metric():
    """Fixture for a simple metric."""
    from sklearn.metrics import accuracy_score
    
    class SimpleMetric(BaseMetric):
        def __init__(self):
            super().__init__(name="accuracy")
        
        def compute(self, y_true, y_pred, **kwargs):
            return accuracy_score(y_true, y_pred)
    
    return SimpleMetric()
