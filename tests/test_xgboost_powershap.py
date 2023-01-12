__author__ = "Jeroen Van Der Donckt, Jarne Verhaeghe"

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from powershap import PowerShap

from .conftest import dummy_classification, dummy_regression


def test_xgboost_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=XGBClassifier(n_estimators=250, verbosity=0), power_iterations=50, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_xgboost_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=XGBRegressor(n_estimators=250, verbosity=0), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_xgboost_handle_nans(dummy_classification):
    X, y = dummy_classification
    X.iloc[:2] = None
    X["nan_col"] = None
    assert np.any(pd.isna(X))
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=XGBClassifier(n_estimators=250, verbosity=0, learning_rate=2), power_iterations=10
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])
