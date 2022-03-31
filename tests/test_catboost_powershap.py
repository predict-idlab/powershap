__author__ = "Jeroen Van Der Donckt"

import numpy as np
import pandas as pd

from powershap import PowerSHAP
from .conftest import dummy_classification, dummy_regression

from catboost import CatBoostClassifier, CatBoostRegressor


def test_catboost_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerSHAP(
        model=CatBoostClassifier(n_estimators=250, verbose=0),
        power_iterations=15,
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_catboost_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerSHAP(
        model=CatBoostRegressor(n_estimators=250, verbose=0),
        power_iterations=15,
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_catboost_handle_nans(dummy_classification):
    X, y = dummy_classification
    X["nan_col"] = None
    assert np.any(pd.isna(X))
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerSHAP(
        model=CatBoostClassifier(n_estimators=250, verbose=0),
        power_iterations=15,
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])
