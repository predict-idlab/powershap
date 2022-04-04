__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

import shap
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from abc import ABC

from typing import Any, Callable


class ShapExplainer(ABC):
    """Interface class for a (POWERshap explainer class."""

    def __init__(
        self,
        model: Any,
    ):
        """Create a POWERshap explainer instance.

        Parameters
        ----------
        model: Any
            The  model from which powershap will use its shap values to perform feature
            selection.

        """
        assert self.supports_model(model)
        self.model = model

    # Should be implemented by subclass
    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs
    ) -> np.array:
        raise NotImplementedError

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        return validate_data(X, y, **kwargs)

    # Should be implemented by subclass
    @staticmethod
    def supports_model(model) -> bool:
        """Check whether the POWERshap explainer supports the given model.

        Parameters
        ----------
        model: Any
            The model.

        Returns
        -------
        bool
            True if the POWERshap expliner supports the given model, otherwise False.

        """
        raise NotImplementedError

    def explain(
        self,
        X: pd.DataFrame,
        y: np.array,
        loop_its: int,
        val_size: float,
        stratify: np.array = None,
        random_seed_start: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """Get the shap values,

        Parameters
        ----------
        X: pd.DataFrame
            The features.
        y: np.array
            The labels.
        loop_its: int
            The number of iterations to fit the model with random state and random
            feature.
        val_size: float
            The fractional size of the validation set. Should be a float between ]0,1[.
        stratify: np.array, optional
            The array used to create a stratified train_test_split. By default None.
        random_seed_start: int, optional
            The random seed to start the iterations with. By default 0.
        **kwargs: dict
            The keyword arguments for the fit method.
        """
        random_col_name = "random_uniform_feature"
        assert not random_col_name in X.columns

        X = X.copy(deep=True)

        shaps = []  # TODO: pre-allocate for efficiency

        for i in tqdm(range(loop_its)):
            npRandomState = RandomState(i + random_seed_start)

            # Add uniform random feature to X
            random_uniform_feature = npRandomState.uniform(-1, 1, len(X))
            X["random_uniform_feature"] = random_uniform_feature

            # Perform train-test split
            train_idx, val_idx = train_test_split(
                np.arange(len(X)),
                test_size=val_size,
                random_state=i,
                stratify=stratify,
            )

            X_train = X.iloc[np.sort(train_idx)]
            X_val = X.iloc[np.sort(val_idx)]
            Y_train = y[np.sort(train_idx)]
            Y_val = y[np.sort(val_idx)]

            Shap_values = self._fit_get_shap(
                X_train=X_train.values,
                Y_train=Y_train,
                X_val=X_val.values,
                Y_val=Y_val,
                random_seed=i + random_seed_start,
                **kwargs,
            )

            Shap_values = np.abs(Shap_values)

            # TODO: consider to convert to even float16?
            shaps += [np.mean(Shap_values, axis=0).astype("float64")]

        shaps = np.array(shaps)

        return pd.DataFrame(data=shaps, columns=X_train.columns.values)


### CATBOOST

from catboost import CatBoostRegressor, CatBoostClassifier


class CatboostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        supported_models = [CatBoostRegressor, CatBoostClassifier]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        if np.any(pd.isna(X)):
            from sklearn.impute import SimpleImputer

            # https://github.com/scikit-learn/scikit-learn/issues/16426
            all_nans = pd.isna(X).all(axis=0)
            X_ = np.zeros(X.shape)
            X_[:, ~all_nans] = SimpleImputer().fit_transform(X)
            if isinstance(X, pd.DataFrame):
                X_ = pd.DataFrame(X_, columns=X.columns)  # , index=X.index)
            _, y = validate_data(X_, y, **kwargs)
            X = np.asarray(X.values if isinstance(X, pd.DataFrame) else X)
            return X, y
        # If there are no nans we can perform the default validation
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs
    ) -> np.array:
        # Fit the model
        PowerSHAP_model = self.model.copy().set_params(random_seed=random_seed)
        PowerSHAP_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerSHAP_model)
        return C_explainer.shap_values(X_val)


### RANDOMFOREST


class EnsembleExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        # TODO: these first 2 require extra checks on the base_estimator
        # from sklearn.ensemble._weight_boosting import BaseWeightBoosting
        # from sklearn.ensemble._bagging import BaseBagging
        from sklearn.ensemble._forest import ForestRegressor, ForestClassifier
        from sklearn.ensemble._gb import BaseGradientBoosting

        supported_models = [ForestRegressor, ForestClassifier, BaseGradientBoosting]
        return isinstance(model, tuple(supported_models))

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs
    ) -> np.array:
        from sklearn.base import clone

        # Fit the model
        PowerSHAP_model = clone(self.model).set_params(random_state=random_seed)
        PowerSHAP_model.fit(X_train, Y_train)
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerSHAP_model)
        try:
            return C_explainer.shap_values(X_val)[1]
        except:
            return C_explainer.shap_values(X_val)


### LINEAR


class LinearExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
        from sklearn.linear_model._stochastic_gradient import BaseSGD

        supported_models = [LinearClassifierMixin, LinearModel, BaseSGD]
        return isinstance(model, tuple(supported_models))

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs
    ) -> np.array:
        from sklearn.base import clone

        # Fit the model
        try:
            PowerSHAP_model = clone(self.model).set_params(random_state=random_seed)
        except:
            PowerSHAP_model = clone(self.model)
        PowerSHAP_model.fit(X_train, Y_train)

        # Calculate the shap values
        C_explainer = shap.explainers.Linear(PowerSHAP_model, X_train)
        return C_explainer.shap_values(X_val)


### DEEP LEARNING


class DeepLearningExplainer(ShapExplainer):
    # TODO
    @staticmethod
    def supports_model(model) -> bool:
        import tensorflow as tf

        # import torch  ## TODO: do we support pytorch??

        supported_models = [tf.keras.Model]  # , torch.nn.Module]
        return isinstance(model, tuple(supported_models))

    def _fit_get_shap(
        self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs
    ) -> np.array:
        # Fit the model
        PowerSHAP_model = self.model
        PowerSHAP_model.compile(
            loss=kwargs["loss"],
            optimizer=kwargs["optimizer"],
            metrics=[kwargs["nn_metric"]],
        )
        _ = PowerSHAP_model.fit(
            X_train,
            Y_train,
            batch_size=kwargs["batch_size"],
            epochs=kwargs["epochs"],
            validation_data=(X_val, Y_val),
            verbose=False,
        )
        # Calculate the shap values
        C_explainer = shap.DeepExplainer(PowerSHAP_model, X_train)
        return C_explainer.shap_values(X_val)[0]  # TODO: why [0]?
