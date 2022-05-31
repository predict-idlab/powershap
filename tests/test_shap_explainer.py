import pytest
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor

from powershap.shap_wrappers import ShapExplainerFactory
from powershap.shap_wrappers.shap_explainer import (
    LinearExplainer,
    CatboostExplainer,
    LGBMExplainer,
    EnsembleExplainer,
    DeepLearningExplainer,
)


def test_get_linear_explainer():
    from sklearn.linear_model import (
        LogisticRegression,
        LogisticRegressionCV,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
    )

    model_classes = [
        # Classifiers
        LogisticRegression,
        LogisticRegressionCV,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        # Regressors
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
    ]

    for model_class in model_classes:
        explainer = ShapExplainerFactory.get_explainer(model_class())
        assert isinstance(explainer, LinearExplainer)


def test_get_catboost_explainer():
    from catboost import CatBoostClassifier, CatBoostRegressor

    model_classes = [CatBoostClassifier, CatBoostRegressor]

    for model_class in model_classes:
        explainer = ShapExplainerFactory.get_explainer(model_class())
        assert isinstance(explainer, CatboostExplainer)


def test_get_lightgbm_explainer():
    from lightgbm import LGBMClassifier, LGBMRegressor

    model_classes = [LGBMClassifier, LGBMRegressor]

    for model_class in model_classes:
        explainer = ShapExplainerFactory.get_explainer(model_class())
        assert isinstance(explainer, LGBMExplainer)


def test_get_ensemble_explainer():
    from sklearn.ensemble import (
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        # HistGradientBoostingClassifier,
        RandomForestRegressor,
        AdaBoostRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        # HistGradientBoostingRegressor,
    )

    model_classes = [
        RandomForestClassifier,
        # AdaBoostClassifier,  # Requires extra check on the base_estimator
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        # HistGradientBoostingClassifier,
        RandomForestRegressor,
        # AdaBoostRegressor,  # Requires extra check on the base_estimator
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        # HistGradientBoostingRegressor,
    ]

    for model_class in model_classes:
        explainer = ShapExplainerFactory.get_explainer(model_class())
        assert isinstance(explainer, EnsembleExplainer)


def test_get_deep_learning_explainer():
    import tensorflow as tf

    explainer = ShapExplainerFactory.get_explainer(tf.keras.Sequential())
    assert isinstance(explainer, DeepLearningExplainer)


def test_value_error_get_explainer():
    with pytest.raises(ValueError):
        ShapExplainerFactory.get_explainer(None)
