__author__ = "Jeroen Van Der Donckt"

from distutils.log import Log
import numpy as np
import pandas as pd

from powershap import PowerSHAP
from .conftest import dummy_classification, dummy_regression

from sklearn.linear_model import LogisticRegression, LinearRegression


def test_logistic_regr_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerSHAP(
        model=LogisticRegression(),
        power_iterations=15, automatic=False,
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )


def test_linear_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerSHAP(
        model=LinearRegression(),
        power_iterations=15, automatic=False,
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert (
        sum([c.startswith("informative") for c in selected_feats.columns])
        == n_informative
    )
