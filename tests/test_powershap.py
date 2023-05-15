__author__ = "Jeroen Van Der Donckt"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from powershap import PowerShap

from .conftest import dummy_classification, dummy_regression

import hashlib

### DEFAULT MODEL & AUTOMATIC MODE


def test_default_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(power_iterations=15, automatic=False)
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostClassifier)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


def test_default_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    selector = PowerShap(power_iterations=15, automatic=False)
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostRegressor)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


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
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


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
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative


### INPUT FEATURE NAMES


def test_powershap_dataframe(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,
        automatic=False,
        show_progress=False,
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
        model=CatBoostClassifier(n_estimators=10, verbose=0), power_iterations=5, automatic=False
    )

    assert isinstance(X, np.ndarray)
    selector.fit(X, y)
    assert not hasattr(selector, "feature_names_in_")
    selected_feats = selector.transform(X)
    assert isinstance(selected_feats, np.ndarray)


### STRATIFY & GROUPS


def test_powershap_stratify_constructor(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,
        automatic=False,
        stratify=True,
    )

    assert selector.stratify is True
    assert selector.cv is None

    selector.fit(X, y)


def test_powershap_stratify_fit(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0), power_iterations=5, automatic=False
    )

    assert selector.cv is None

    selector.fit(X, y, stratify=y)


def test_powershap_groups_fit(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0), power_iterations=5, automatic=False
    )

    assert selector.cv is None

    selector.fit(X, y, groups=np.random.randint(0, 3, size=len(X)))


def test_powershap_stratify_constructor_groups_fit(dummy_classification):
    X, y = dummy_classification

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,
        automatic=False,
        stratify=True,
    )

    assert selector.cv is None

    selector.fit(X, y, groups=np.random.randint(0, 3, size=len(X)))


### CROSS-VALIDATION


def test_powershap_cv_kfold(dummy_classification):
    from sklearn.model_selection import KFold

    X, y = dummy_classification

    cv = KFold(3)

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,  # more than cv
        automatic=False,
        cv=cv,
    )

    assert selector.cv is not None

    selector.fit(X, y)


def test_powershap_cv_groupkfold(dummy_classification):
    from sklearn.model_selection import GroupKFold

    X, y = dummy_classification

    cv = GroupKFold(3)

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,  # more than cv
        automatic=False,
        cv=cv,
    )

    assert selector.cv is not None

    selector.fit(X, y, groups=np.random.randint(0, 3, size=len(X)))


def test_powershap_cv_stratifiedgroupkfold(dummy_classification):
    from sklearn.model_selection import StratifiedGroupKFold

    X, y = dummy_classification

    cv = StratifiedGroupKFold(3)

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,  # more than cv
        automatic=False,
        cv=cv,
    )

    assert selector.cv is not None

    selector.fit(X, y, groups=np.random.randint(0, 3, size=len(X)))


def test_powershap_cv_groupshufflesplit(dummy_classification):
    from sklearn.model_selection import GroupShuffleSplit

    X, y = dummy_classification

    cv = GroupShuffleSplit(3)

    selector = PowerShap(
        model=CatBoostClassifier(n_estimators=10, verbose=0),
        power_iterations=5,  # more than cv
        automatic=False,
        cv=cv,
    )

    assert selector.cv is not None

    selector.fit(X, y, groups=np.random.randint(0, 3, size=len(X)))


def test_no_mutate_df(dummy_classification):
    """Ensure that powershap fit doesn't mutate an input pandas dataframe.
    """
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    assert isinstance(X, pd.DataFrame)

    hash1 = hashlib.sha1(pd.util.hash_pandas_object(X).values).hexdigest()

    selector = PowerShap(power_iterations=15, automatic=False)
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostClassifier)
    selected_feats = selector.transform(X)

    assert len(selected_feats.columns) >= n_informative
    assert sum([c.startswith("informative") for c in selected_feats.columns]) == n_informative

    hash2 = hashlib.sha1(pd.util.hash_pandas_object(X).values).hexdigest()

    assert hash1 == hash2


def test_no_mutate_numpy(dummy_classification):
    """Ensure that powershap fit doesn't mutate an input numpy array.
    """
    X, y = dummy_classification

    X = X.to_numpy()

    hash1 = hashlib.sha1(pd.util.hash_pandas_object(pd.DataFrame(X)).values).hexdigest()

    selector = PowerShap(power_iterations=15, automatic=False)
    assert selector.model is None

    selector.fit(X, y)
    assert isinstance(selector.model, CatBoostClassifier)
    selected_feats = selector.transform(X)
    assert selected_feats.shape[1] > 0

    hash2 = hashlib.sha1(pd.util.hash_pandas_object(pd.DataFrame(X)).values).hexdigest()

    assert hash1 == hash2
