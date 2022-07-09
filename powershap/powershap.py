__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

import warnings
import sklearn
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import BaseCrossValidator

from .shap_wrappers import ShapExplainerFactory

from .utils import powerSHAP_statistical_analysis


class PowerShap(SelectorMixin, BaseEstimator):
    """
    Feature selection based on significance of shap values.

    """

    def __init__(
        self,
        model=None,
        power_iterations: int = 10,
        power_alpha: float = 0.01,
        val_size: float = 0.2,
        power_req_iterations: float = 0.99,
        include_all: bool = False,
        automatic: bool = True,
        force_convergence: bool = False,
        limit_convergence_its: int = 0,
        limit_automatic: int = 10,
        limit_incremental_iterations: int = 10,
        limit_recursive_automatic: int = 3,
        stratify: bool = False,
        cv: BaseCrossValidator = None,
        show_progress: bool = True,
        verbose: bool = False,
        **fit_kwargs,
    ):
        """
        Create a powershap object.

        Parameters
        ----------
        model: Any, optional
            The model used for for the powershap calculation. The currently supported
            models are; catboost, sklearn tree-based, sklearn linear, and tensorflow
            deep learning models.
            If no model is passed, by default, a catboost model will be used. If the
            data type is of type float, a CatBoostRegressor will be selected, for all
            the other cases a CatBoostClassifier is selected.
            ..note::
                The deep learning model should take |features| + 1 as input size.
                It is thus the user his/her responsibility to account for the added
                random feature, when using deep learning models.
        power_iterations: int, optional
            The number of shuffles and iterations of the power feature selection method,
            ignored when using automatic mode. By default 10.
        power_alpha: float, optional
            The alpha value used for the power-calculation of the used statistical test
            and significance threshold. Should be a float between ]0,1[. By default
            0.01.
        val_size: float, optional
            The fractional size of the validation set. Should be a float between ]0,1[.
            By default 0.2.
        power_req_iterations: float, optional
            The fractional power percentage for the required iterations calculation. By
            default 0.95.
        include_all: bool
            Flag indicating whether all features should be analyzed or only those with a
            threshold of `power_alpha`.
        automatic: bool, optional
            If True, the powershap will first calculate the required iterations by using
            10 iterations and then restart using the required iterations for
            `power_iterations`. By default False.
        force_convergence: bool, optional
            Only used for automatic mode. If True, powershap will continue delete
            the found relevant features and rerun until no more relevant features are found.
            This is especially useful in high-dimensional datasets
        limit_convergence_its: int, optional
            The number of maximum allowed recursions when `force_convergence` is True. By
            default 0, meaning that no limit is applied. A limit_convergence_its of 1 suggests 
            only executing one convergence recursion 
            after a single full automatic PowerShap execution. 
        limit_automatic: int, optional
            The number of maximum allowed iterations when `automatic` is True. By
            default None, meaning that no limit is applied.
        limit_incremental_iterations: int, optional
            If the required iterations exceed `limit_automatic` in automatic mode, add
            `limit_incremental_iterations` iterations and re-evaluate. By default 10.
        limit_recursive_automatic: int, optional
            The number of maximum allowed times that `limit_incremental_iterations`
            iterations are added. This restricts the amount of powershap recursion. By
            default 3.
        stratify: bool, optional
            Whether to create a stratified train_test_split (based on the `y` that is
            given to the `.fit` method). By default False.
            ..note::
                If you want to pass a specific array as stratify (that is not `y`), you
                can pass it as `stratify` argument to the `.fit` method.
        cv: BaseCrossValidator, optional
            The cross-validator to use. By default None.
            This cross-validator should have a `.split` method which yields
            (train_idx, test_idx) tuples. The arguments of the `.split` method should be
            X, y, groups. This splitter will be wrapped to yield infinite splits.
            ..note::
                If the given coss validator has no `random_state` argument, the same
                splits will be used multiple times in the powershap iterations. This
                may lead to overfitting on the cross-validation splits (and thus
                selection of non-informative variables).
        show_progress: bool, optional
            Flag indicating whether progress of the powershap iterations should be
            shown. By default True.
        verbose: bool, optional
            Flag indicating whether verbose console output should be shown. By default
            False.
        **fit_kwargs: dict
            Keyword arguments for fitting the model.
            ..note::
                For a deep learning model, the following keyword arguments are required:
                "epochs", "optimizer", "batch_size", "nn_metric", "loss"

        """
        self.model = model
        self.power_iterations = power_iterations
        self.power_alpha = power_alpha
        self.val_size = val_size
        self.power_req_iterations = power_req_iterations
        self.include_all = include_all
        self.automatic = automatic
        self.force_convergence = force_convergence
        self.limit_convergence_its = limit_convergence_its
        self.limit_automatic = limit_automatic
        self.limit_incremental_iterations = limit_incremental_iterations
        self.limit_recursive_automatic = limit_recursive_automatic
        self.stratify = stratify
        self.show_progress = show_progress
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs

        def _infinite_splitter(cv):
            """Infinite yields for the given splitter.
            If the splitter is exhausted, it will be reset and restarted.
            """
            from copy import deepcopy
            cv = deepcopy(cv)
            splitter = None 
            random_state = 0
            def split(X, y=None, groups=None):
                nonlocal splitter, random_state
                if splitter is None:
                    if hasattr(cv, "random_state"):  # Update random state
                        cv.__setattr__("random_state", random_state)
                        random_state += 1
                    splitter = cv.split(X, y=y, groups=groups)
                while True:
                    try:
                        yield next(splitter)
                    except StopIteration:
                        if hasattr(cv, "random_state"):  # Update random state
                            cv.__setattr__("random_state", random_state)
                            random_state += 1
                        splitter = cv.split(X, y=y, groups=groups)
                        yield next(splitter)
            return split
     
        if cv is not None:
            self.cv = _infinite_splitter(cv)
        else:
            self.cv = None

        if model is not None:
            self._explainer = ShapExplainerFactory.get_explainer(model=model)

    @staticmethod
    def _get_default_model(y: np.ndarray):
        from catboost import CatBoostClassifier, CatBoostRegressor

        assert isinstance(y, np.ndarray)
        dtype = y.dtype
        if np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.integer):
            return CatBoostRegressor(
                n_estimators=250,
                od_type="Iter",
                od_wait=25,
                use_best_model=True,
                verbose=0,
            )
        if np.issubdtype(dtype, np.integer) and len(np.unique(y.ravel())) >= 5:
            warnings.warn(
                "Classifying although there are >= 5 integers in the labels.",
                UserWarning,
            )
        return CatBoostClassifier(
            n_estimators=250, od_type="Iter", od_wait=25, use_best_model=True, verbose=0
        )

    def _log_feature_names_sklean_v0(self, X):
        """Log the feature names if we have sklearn 0.x"""
        assert sklearn.__version__.startswith("0.")
        feature_names = np.asarray(X.columns) if hasattr(X, "columns") else None
        if feature_names is not None and len(feature_names) > 0:
            # Check if all feature names of type string
            types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
            if len(types) > 1 or types[0] != "str":
                feature_names = None
                warnings.warn(
                    "Feature names only support names that are all strings.",
                    UserWarning,
                )

        if feature_names is not None and len(feature_names) > 0:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            # Delete the attribute when the estimator is fitted on a new dataset that
            # has no feature names.
            delattr(self, "feature_names_in_")

    def _print(self, *values):
        """Helper method for printing if `verbose` is set to True."""
        if self.verbose:
            print(*values)

    def _automatic_fit(
        self, X, y, processed_shaps_df, loop_its, stratify, groups, shaps_df, **kwargs
    ):
        if not any(processed_shaps_df.p_value < self.power_alpha):
            # There is no feature found yet...
            self._print("No features selected after 10 automatic iterations!")
            # Return already as more iterations will only result in including less
            # features
            return processed_shaps_df

        max_iterations = int(
            np.ceil(
                processed_shaps_df[processed_shaps_df.p_value < self.power_alpha][
                    str(self.power_req_iterations) + "_power_its_req"
                ].max()
            )
        )

        max_iterations_old = loop_its
        recurs_counter = 0

        if max_iterations <= max_iterations_old:
            self._print(
                f"{loop_its} iterations were already sufficient as only",
                f"{max_iterations} iterations were required for the current ",
                f"power_alpha = {self.power_alpha}.",
            )

        while (
            max_iterations > max_iterations_old
            # and max_iterations < limit_automatic
            and recurs_counter < self.limit_recursive_automatic
        ):

            shaps_df_recursive: pd.DataFrame = None
            if max_iterations - max_iterations_old > self.limit_automatic:
                self._print(
                    f"Automatic mode: powershap requires {max_iterations} ",
                    "iterations; The extra required iterations exceed the limit_automatic ",
                    "threshold. Powershap will add ",
                    f"{self.limit_incremental_iterations} powershap iterations and ",
                    "re-evaluate.",
                )

                shaps_df_recursive = self._explainer.explain(
                    X=X,
                    y=y,
                    loop_its=self.limit_incremental_iterations,
                    val_size=self.val_size,
                    stratify=stratify,
                    groups=groups,
                    cv_split=self.cv,  # pass the wrapped cv split function
                    random_seed_start=max_iterations_old,
                    show_progress=self.show_progress,
                    **kwargs,
                )

                max_iterations_old = (
                    max_iterations_old + self.limit_incremental_iterations
                )

            else:
                self._print(
                    f"Automatic mode: Powershap requires {max_iterations} "
                    f"iterations; Adding {max_iterations-max_iterations_old} ",
                    "powershap iterations.",
                )

                shaps_df_recursive = self._explainer.explain(
                    X=X,
                    y=y,
                    loop_its=max_iterations - max_iterations_old,
                    val_size=self.val_size,
                    stratify=stratify,
                    groups=groups,
                    cv_split=self.cv,  # pass the wrapped cv split function
                    random_seed_start=max_iterations_old,
                    show_progress=self.show_progress,
                    **kwargs,
                )

                max_iterations_old = max_iterations

            shaps_df = pd.concat([shaps_df, shaps_df_recursive])

            processed_shaps_df = powerSHAP_statistical_analysis(
                shaps_df,
                self.power_alpha,
                self.power_req_iterations,
                include_all=self.include_all,
            )

            if not any(processed_shaps_df.p_value < self.power_alpha):
                # There is no feature found yet...
                self._print("No features selected after 10 automatic iterations!")
                # Return already as more iterations will only result in including less
                # features
                return processed_shaps_df

            max_iterations = int(
                np.ceil(
                    processed_shaps_df[processed_shaps_df.p_value < self.power_alpha][
                        str(self.power_req_iterations) + "_power_its_req"
                    ].max()
                )
            )

            recurs_counter += 1

        return processed_shaps_df

    def fit(self, X, y, stratify=None, groups=None, **kwargs):
        """Fit the powershap feature selector.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y: array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        stratify: array-like of shape (n_samples,), optional
            Array that will be used to perform stratified train-test splits. By default
            None.
            Note: if None, than `y` will be used as `stratify` if the stratify flag of
            the object is True.
        groups: array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. By default None.

        """
        if stratify is None and self.stratify:
            # Set stratify to y, if no stratify is given and self.stratify is True
            stratify = y
        
        # kwargs take precedence over fit_kwargs
        kwargs = {**self.fit_kwargs, **kwargs}

        if self.model is None:
            # If no model is passed to the constructor -> select the default catboost
            # model
            self.model = PowerShap._get_default_model(np.asarray(y))
            self._explainer = ShapExplainerFactory.get_explainer(self.model)

        if sklearn.__version__.startswith("0."):
            # Log the feature names if we have sklearn 0.x
            self._log_feature_names_sklean_v0(X)
        # Perform the necessary sklearn checks -> X and y are both ndarray
        # Logs the feature names as well (in self.feature_names_in_ in sklearn 1.x)
        X, y = self._explainer._validate_data(
            self._validate_data, X, y, multi_output=True
        )

        self._print("Starting powershap")

        X = pd.DataFrame(data=X, columns=list(range(X.shape[1])))

        loop_its = self.power_iterations
        if self.automatic:
            loop_its = 10
            self._print(
                "Automatic mode enabled: Finding the minimal required powershap",
                f"iterations for significance of {self.power_alpha}.",
            )

        shaps_df = self._explainer.explain(
            X=X,
            y=y,
            loop_its=loop_its,
            val_size=self.val_size,
            stratify=stratify,
            groups=groups,
            cv_split=self.cv,  # pass the wrapped cv split function
            show_progress=self.show_progress,
            **kwargs,
        )

        processed_shaps_df = powerSHAP_statistical_analysis(
            shaps_df,
            self.power_alpha,
            self.power_req_iterations,
            include_all=self.include_all,
        )

        if self.automatic:

            processed_shaps_df = self._automatic_fit(
                X=X,
                y=y,
                processed_shaps_df=processed_shaps_df,
                loop_its=loop_its,
                stratify=stratify,
                groups=groups,
                shaps_df=shaps_df,
                **kwargs,
            )

            # Continue powershap until no more informative features are found
            if self.force_convergence:
                self._print("Forcing convergence.")

                converge_df = processed_shaps_df.copy()

                significant_cols = np.array(
                    converge_df[converge_df.p_value < self.power_alpha].index.values
                )

                # If limit_convergence_its is zero, the convergence mode does not have a limit. If not, the 
                # While loop condition is recalculated every while loop iteration.
                if self.limit_convergence_its > 0:
                    current_converge_recursions = 0
                    while_convergence_bool = current_converge_recursions < self.limit_convergence_its
                else:
                    while_convergence_bool = True

                while((len(converge_df[converge_df.p_value < self.power_alpha]) > 0) & (while_convergence_bool)):
                    self._print("Rerunning powershap for convergence. ")
                    converge_shaps_df = self._explainer.explain(
                        X=X.drop(
                            columns=X.columns.values[significant_cols.astype(np.int32)]
                        ),
                        y=y,
                        loop_its=loop_its,
                        val_size=self.val_size,
                        stratify=stratify,
                        groups=groups,
                        cv_split=self.cv,  # pass the wrapped cv split function
                        show_progress=self.show_progress,
                        **kwargs,
                    )

                    converge_df = powerSHAP_statistical_analysis(
                        converge_shaps_df,
                        self.power_alpha,
                        self.power_req_iterations,
                        include_all=self.include_all,
                    )

                    converge_df = self._automatic_fit(
                        X=X.drop(
                            columns=X.columns.values[significant_cols.astype(np.int32)]
                        ),
                        y=y,
                        processed_shaps_df=converge_df,
                        loop_its=loop_its,
                        stratify=stratify,
                        groups=groups,
                        converge_shaps_df=converge_shaps_df,
                        shaps_df=converge_shaps_df,
                        **kwargs,
                    )

                    significant_cols = np.append(
                        significant_cols,
                        converge_df[
                            converge_df.p_value < self.power_alpha
                        ].index.values,
                    )

                    processed_shaps_df.loc[
                        converge_df[converge_df.p_value < self.power_alpha].index.values
                    ] = converge_df[converge_df.p_value < self.power_alpha]

                    
                    if self.limit_convergence_its > 0:
                        current_converge_recursions += 1
                        print(current_converge_recursions)
                        while_convergence_bool = current_converge_recursions < self.limit_convergence_its

                        if not while_convergence_bool:
                            self._print("Convergence limit reached: Stopping convergence mode.")


                processed_shaps_df.loc[converge_df.index.values] = converge_df


        self._print("Done!")

        ## Set the p-values property (used in the transform function)
        # Remove the random feature (legit features have int index)
        sub_df = processed_shaps_df[
            processed_shaps_df.index.map(lambda x: isinstance(x, int))
        ]
        # Sort to have original order again
        sub_df = sub_df.sort_index()
        self._p_values = sub_df.p_value.values

        ## Store the processed_shaps_df in the object
        self._processed_shaps_df = processed_shaps_df
        if hasattr(self, "feature_names_in_"):
            self._processed_shaps_df.index = [
                self.feature_names_in_[i] if isinstance(i, int) else i
                for i in processed_shaps_df.index.values
            ]

        # It is convention to return self
        return self

    # This is the only method that needs to be implemented to serve the transform
    # functionality
    def _get_support_mask(self):
        # Select the significant features
        return self._p_values < self.power_alpha

    def transform(self, X):
        check_is_fitted(self, ["_processed_shaps_df", "_p_values", "_explainer"])
        if hasattr(self, "feature_names_in_") and isinstance(X, pd.DataFrame):
            assert np.all(X.columns.values == self.feature_names_in_)
            return pd.DataFrame(
                super().transform(X),
                columns=self.feature_names_in_[self._get_support_mask()],
            )
        return super().transform(X)

    def _more_tags(self):
        return self._explainer._get_more_tags()
