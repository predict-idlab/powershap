__author__ = "Jeroen Van Der Donckt"

import numpy as np
import pandas as pd

from powershap import PowerShap
from .conftest import dummy_classification, dummy_regression

from catboost import CatBoostClassifier, CatBoostRegressor

### DEFAULT MODEL & AUTOMATIC MODE


def test_default_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        power_iterations=15,
        automatic=False,
    )
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostClassifier)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )


def test_default_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        power_iterations=15,
        automatic=False,
    )
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostRegressor)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )


def test_default_class_automatic_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap()
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostClassifier)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )


def test_default_regr_automatic_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap()
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostRegressor)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )


### INPUT FEATURE NAMES


def test_powershap_dataframe(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,
        automatic=False,
    )

    assert isinstance(X, pd.DataFrame)
    selector.fit(X, y)
    assert hasattr(selector, "feature_names_in_")
    assert np.all(selector.feature_names_in_ == X.columns)
    selected_feats = selector.transform(X)
    assert isinstance(selected_feats, pd.DataFrame)


def test_powershap_array(dummy_classification):
    X, y = dummy_classification
    X = X.values

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,
        automatic=False,
    )

    assert isinstance(X, np.ndarray)
    selector.fit(X, y)
    assert not hasattr(selector, "feature_names_in_")
    selected_feats = selector.transform(X)
    assert isinstance(selected_feats, np.ndarray)
