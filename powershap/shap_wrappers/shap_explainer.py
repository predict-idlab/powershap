__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

import gc
import warnings
from abc import ABC
from copy import copy
from typing import Any, Callable

import numpy as np
import pandas as pd
import shap
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class ShapExplainer(ABC):
    """Interface class for a (POWERshap explainer class."""

    def __init__(self, model: Any):
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
    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
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
        groups: np.array = None,
        cv_split: Callable = None,
        random_seed_start: int = 0,
        show_progress: bool = False,
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
            The number of iterations to fit the model with random state and random feature.
        val_size: float
            The fractional size of the validation set. Should be a float between ]0,1[.
        stratify: np.array, optional
            The array used to create a stratified train_test_split. By default None.
        groups: np.array, optional
            The group labels for the samples used while splitting the dataset into
            train/test set. By default None.
        cv_split: Callable, optional
            The function used to create a cross-validation split. By default None.
            This function is a wrapper of the BaseCrossValidator that will yield
            infinite amount of splits. The arguments of this function are X, y, groups.
        random_seed_start: int, optional
            The random seed to start the iterations with. By default 0.
        **kwargs: dict
            The keyword arguments for the fit method.
        """
        random_col_name = "random_uniform_feature"

        shaps = []  # TODO: pre-allocate for efficiency

        cv_splitter = None
        if cv_split is not None:
            # Pass the stratify array as y if stratify is not None
            y_ = stratify if stratify is not None else y
            cv_splitter = cv_split(X, y=y_, groups=groups)

        iterations = tqdm(range(loop_its)) if show_progress else range(loop_its)
        for i in iterations:
            random_seed = i + random_seed_start
            npRandomState = RandomState(random_seed)

            # Add uniform random feature to X.
            X[random_col_name] = npRandomState.uniform(-1, 1, len(X))

            ### A) Split using the wrapped cross-validation splitter
            if cv_splitter is not None:
                train_idx, val_idx = next(cv_splitter)
            ### B) Perform train-test split when no cross-validation splitter is passed
            elif groups is None:
                # stratify may be None or not None
                train_idx, val_idx = train_test_split(
                    np.arange(len(X)), test_size=val_size, random_state=i, stratify=stratify
                )
            elif stratify is None:
                # groups may be None or not None
                from sklearn.model_selection import GroupShuffleSplit

                train_idx, val_idx = next(
                    GroupShuffleSplit(random_state=i, n_splits=1, test_size=val_size).split(
                        X, y, groups=groups
                    )
                )
            else:
                # stratify and groups are both not None
                try:
                    from sklearn.model_selection import StratifiedGroupKFold

                    train_idx, val_idx = next(
                        StratifiedGroupKFold(
                            shuffle=True, random_state=i, n_splits=int(1 / val_size)
                        ).split(X, y, groups=groups)
                    )
                except Exception:
                    warnings.warn(
                        "Did not find StratifiedGroupKFold in sklearn install, "
                        + "this is only supported in sklearn 1.x.",
                        UserWarning,
                    )
                    train_idx, val_idx = train_test_split(
                        np.arange(len(X)), test_size=val_size, random_state=i, stratify=stratify
                    )

            X_train = X.iloc[np.sort(train_idx)].to_numpy()
            X_val = X.iloc[np.sort(val_idx)].to_numpy()
            Y_train = y[np.sort(train_idx)]
            Y_val = y[np.sort(val_idx)]

            Shap_values = self._fit_get_shap(
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                random_seed=random_seed,
                **kwargs,
            )

            # If our data is large, we don't want to hold two copies in memory
            # at once, so we manually release them before the next assignment.
            X_val_shape = X_val.shape
            del X_train, X_val, Y_train, Y_val

            Shap_values = np.abs(Shap_values)

            if len(np.shape(Shap_values)) > 2:
                # (expected) SHAPE: (n_samples, n_features, n_outputs)
                assert len(np.shape(Shap_values)) == 3, "Shap values should be 3D"
                # NOTE: depending on the shap library, the n_output axis location can vary
                #       so we need to find it dynamically
                axis = np.where(np.array(Shap_values.shape) == len(np.unique(y)))[0]
                assert len(axis == 1), "Shap values should have only one axis with n_outputs"
                axis = axis[0]

                # in case of multi-output, we take the max of the outputs as the shap value
                Shap_values = np.max(Shap_values, axis=axis)
                # new shape = (n_samples, n_features)

            # TODO: consider to convert to even float16?
            assert np.shape(Shap_values) == X_val_shape, "Shap vals should have same shape as X_val"
            Shap_values = np.mean(Shap_values, axis=0).astype("float32")
            # new shape = (n_features,)

            shaps += [Shap_values]

            # Python's reference-counting garbage collection has a hard time
            # cleaning up after performing some of the work above. Running a
            # manual garbage collection here keeps memory usage from increasing
            # on every iteration.
            gc.collect()

        shaps = np.array(shaps)

        return pd.DataFrame(data=shaps, columns=X.columns.values)

    def _get_more_tags(self):
        return {}


### CATBOOST

from catboost import CatBoostClassifier, CatBoostRegressor


class CatboostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        supported_models = [CatBoostRegressor, CatBoostClassifier]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        kwargs["force_all_finite"] = False  # catboost allows NaNs and infs in X
        kwargs["dtype"] = None  # allow non-numeric data
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        # Fit the model
        PowerShap_model = self.model.copy().set_params(random_seed=random_seed)
        PowerShap_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerShap_model)
        return C_explainer.shap_values(X_val)

    def _get_more_tags(self):
        return {"allow_nan": True}


### LGBM


class LGBMExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from lightgbm import LGBMClassifier, LGBMRegressor

        supported_models = [LGBMClassifier, LGBMRegressor]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        kwargs["force_all_finite"] = False  # lgbm allows NaNs and infs in X
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        # Fit the model
        # Why we need to use deepcopy and delete LGBM __deepcopy__
        # https://github.com/microsoft/LightGBM/issues/4085
        PowerShap_model = copy(self.model).set_params(random_seed=random_seed)
        PowerShap_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerShap_model)
        return C_explainer.shap_values(X_val)

    def _get_more_tags(self):
        return {"allow_nan": True}


### XGBOOST


class XGBoostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from xgboost import XGBClassifier, XGBRegressor

        supported_models = [XGBClassifier, XGBRegressor]
        return isinstance(model, tuple(supported_models))

    def _validate_data(self, validate_data: Callable, X, y, **kwargs):
        kwargs["force_all_finite"] = False  # xgboost allows NaNs and infs in X
        kwargs["dtype"] = None  # allow non-numeric data
        return super()._validate_data(validate_data, X, y, **kwargs)

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        # Fit the model
        PowerShap_model = copy(self.model).set_params(random_state=random_seed)
        PowerShap_model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerShap_model)
        return C_explainer.shap_values(X_val)

    def _get_more_tags(self):
        return {"allow_nan": True}


### RANDOMFOREST


class EnsembleExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        # TODO: these first 2 require extra checks on the base_estimator
        # from sklearn.ensemble._weight_boosting import BaseWeightBoosting
        # from sklearn.ensemble._bagging import BaseBagging
        from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
        from sklearn.ensemble._gb import BaseGradientBoosting

        # from sklearn.ensemble._hist_gradient_boosting import BaseHistGradientBoosting

        supported_models = [ForestRegressor, ForestClassifier, BaseGradientBoosting]
        return issubclass(type(model), tuple(supported_models))

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        from sklearn.base import clone

        # Fit the model
        PowerShap_model = clone(self.model).set_params(random_state=random_seed)
        PowerShap_model.fit(X_train, Y_train)
        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerShap_model)
        return C_explainer.shap_values(X_val)


### LINEAR


class LinearExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
        from sklearn.linear_model._stochastic_gradient import BaseSGD

        supported_models = [LinearClassifierMixin, LinearModel, BaseSGD]
        return issubclass(type(model), tuple(supported_models))

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        from sklearn.base import clone

        # Fit the model
        try:
            PowerShap_model = clone(self.model).set_params(random_state=random_seed)
        except Exception:
            PowerShap_model = clone(self.model)
        PowerShap_model.fit(X_train, Y_train)

        # Calculate the shap values
        C_explainer = shap.explainers.Linear(PowerShap_model, X_train)
        return C_explainer.shap_values(X_val)


### DEEP LEARNING


class DeepLearningExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        import tensorflow as tf  # ; import torch

        # import torch  ## TODO: do we support pytorch??

        supported_models = [tf.keras.Model]  # , torch.nn.Module]
        return isinstance(model, tuple(supported_models))

    def _fit_get_shap(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        import tensorflow as tf

        # tf.compat.v1.disable_v2_behavior()  # https://github.com/slundberg/shap/issues/2189
        # Fit the model
        PowerShap_model = tf.keras.models.clone_model(self.model)
        metrics = kwargs.get("nn_metric")
        PowerShap_model.compile(
            loss=kwargs["loss"],
            optimizer=kwargs["optimizer"],
            metrics=metrics if metrics is None else [metrics],
            # run_eagerly=True,
        )
        _ = PowerShap_model.fit(
            X_train,
            Y_train,
            batch_size=kwargs["batch_size"],
            epochs=kwargs["epochs"],
            validation_data=(X_val, Y_val),
            verbose=False,
        )
        # Calculate the shap values
        C_explainer = shap.DeepExplainer(PowerShap_model, X_train)
        return C_explainer.shap_values(X_val)
