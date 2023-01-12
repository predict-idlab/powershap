__author__ = "Jeroen Van Der Donckt"

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from powershap import PowerShap

from .conftest import dummy_classification, dummy_regression


def test_ensemble_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=RandomForestClassifier(n_estimators=25), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


def test_ensemble_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(
        model=RandomForestRegressor(n_estimators=25), power_iterations=15, automatic=False
    )

    selector.fit(X, y)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative
