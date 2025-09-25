__author__ = "Jeroen Van Der Donckt, Jarne Verhaeghe"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from powershap import PowerShap

from .conftest import dummy_classification, dummy_regression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def test_pipeline_catboost_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler, CatBoostClassifier(n_estimators=250, verbose=0)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_catboost_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler, CatBoostRegressor(n_estimators=250, verbose=0)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_catboost_handle_nans(dummy_classification):
    X, y = dummy_classification
    X.iloc[:5] = None
    X["nan_col"] = None
    assert np.any(pd.isna(X))
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler, CatBoostClassifier(n_estimators=10, verbose=0)), power_iterations=15
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_catboost_handle_infs(dummy_classification):
    X, y = dummy_classification
    X.iloc[:5] = np.Inf
    X["inf_col"] = np.Inf
    assert np.any(X.isin([np.inf, -np.inf]))
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler, CatBoostClassifier(n_estimators=10, verbose=0)), power_iterations=15
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_catboost_handle_infs_nans(dummy_classification):
    X, y = dummy_classification
    X.iloc[:5] = np.Inf
    X.iloc[5:10] = None
    X["inf_col"] = np.Inf
    X["nan_col"] = None
    assert np.any(X.isin([np.inf, -np.inf]))
    assert np.any(pd.isna(X))
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler, CatBoostClassifier(n_estimators=10, verbose=0)), power_iterations=15
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_catboost_handle_strings(dummy_classification):
    X, y = dummy_classification
    X["cat"] = "miauw"
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)

    selector = PowerShap(
        model=make_pipeline(DummyScaler,CatBoostClassifier(n_estimators=30, verbose=0, cat_features=[X.shape[1] - 1])), 
        power_iterations=15
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])


def test_pipeline_ensemble_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)
    
    selector = PowerShap(
        model=make_pipeline(DummyScaler,RandomForestClassifier(n_estimators=25)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


def test_pipeline_ensemble_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    DummyScaler = FunctionTransformer(lambda x: x)
    
    selector = PowerShap(
        model=make_pipeline(DummyScaler,RandomForestRegressor(n_estimators=25)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative

def test_pipeline_catboost_class_standardscaler_powershap(dummy_classification):
    from sklearn.preprocessing import StandardScaler

    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=make_pipeline(StandardScaler(), CatBoostClassifier(n_estimators=250, verbose=0)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])

def test_pipeline_catboost_class_maxabsscalerr_robustscaler_powershap(dummy_classification):
    from sklearn.preprocessing import MaxAbsScaler, RobustScaler

    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=make_pipeline(MaxAbsScaler(), RobustScaler(), CatBoostClassifier(n_estimators=250, verbose=0)), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) == n_informative
    assert all([c.startswith("informative") for c in selected_feats.columns])
