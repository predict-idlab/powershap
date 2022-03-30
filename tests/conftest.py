__author__ = "Jeroen Van Der Donckt"

import pytest
import pandas as pd

from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def dummy_classification() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=250,
        n_features=10,
        n_informative=4,
        n_repeated=0,
        n_redundant=0,
        random_state=42,
        shuffle=False,
    )
    X = pd.DataFrame(
        X,
        columns=[f"informative_{i}" for i in range(4)]
        + [f"random_{i}" for i in range(6)],
    )
    return X, y


@pytest.fixture
def dummy_regression() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=250,
        n_features=10,
        n_informative=4,
        random_state=42,
        shuffle=False,
    )
    X = pd.DataFrame(
        X,
        columns=[f"informative_{i}" for i in range(4)]
        + [f"random_{i}" for i in range(6)],
    )
    return X, y
