import pytest

from powershap.shap_wrappers import ShapExplainerFactory
from powershap.shap_wrappers.shap_explainer import (
    LinearExplainer,
    CatboostExplainer,
    EnsembleExplainer,
    DeepLearningExplainer,
)


def test_get_linear_explainer():
    from sklearn.linear_model import LogisticRegression

    explainer = ShapExplainerFactory.get_explainer(LogisticRegression())
    assert isinstance(explainer, LinearExplainer)


def test_get_catboost_explainer():
    from catboost import CatBoostClassifier

    explainer = ShapExplainerFactory.get_explainer(CatBoostClassifier())
    assert isinstance(explainer, CatboostExplainer)


def test_get_ensemble_explainer():
    from sklearn.ensemble import RandomForestClassifier

    explainer = ShapExplainerFactory.get_explainer(RandomForestClassifier())
    assert isinstance(explainer, EnsembleExplainer)


def test_get_deep_learning_explainer():
    import tensorflow as tf

    explainer = ShapExplainerFactory.get_explainer(tf.keras.Sequential())
    assert isinstance(explainer, DeepLearningExplainer)


def test_value_error_get_explainer():
    with pytest.raises(ValueError):
        ShapExplainerFactory.get_explainer(None)
