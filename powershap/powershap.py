__author__ = "Jarne Verhaeghe, Jeroen Van Der Donckt"

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from copy import deepcopy
from shap_wrappers import ShapExplainerFactory

from utils import powerSHAP_statistical_analysis


class PowerSHAP(SelectorMixin, BaseEstimator):
    """
    Feature selection based on significance of shap values.

    """

    def __init__(
        self,
        model,
        power_iterations: int = 10,
        power_alpha: float = 0.01,
        val_size: float = 0.2,
        power_req_iterations: float = 0.95,
        include_all: bool = False,
        automatic: bool = False,
        limit_automatic: int = None,
        limit_incremental_iterations: int = 10,
        limit_recursive_automatic: int = 3,
        stratify: bool = False,  # TODO
        verbose: bool = False,
        model_kwargs: dict = None,  # TODO
    ):
        """
        Create a PowerSHAP object.

        Parameters
        ----------
        model
            The model used for for the PowerSHAP calculation.
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
            If True, the PowerSHAP will first calculate the required iterations by using
            10 iterations and then restart using the required iterations for
            `power_iterations`. By default False.
            TODO: dit is dan geen harde limiet (zie hieronder?)
        limit_automatic: int, optional
            The number of maximum allowed iterations when `automatic` is True. By
            default None, meaning that no limit is applied.
        limit_incremental_iterations: int, optional
            If the required iterations exceed `limit_automatic` in automatic mode, add
            `limit_incremental_iterations` iterations and re-evaluate. By default 10.
        limit_recursive_automatic: int, optional
            The number of maximum allowed times that `limit_incremental_iterations`
            iterations are added. This restricts the amount of PowerSHAP recursion. By
            default 3 .
        verbose: bool, optional
            Flag indicating whether verbose console output should be shown. By default
            False.

        """
        self.model = model
        self.power_iterations = power_iterations
        self.power_alpha = power_alpha
        self.val_size = val_size
        self.power_req_iterations = power_req_iterations
        self.include_all = include_all
        self.automatic = automatic
        self.limit_automatic = limit_automatic
        self.limit_incremental_iterations = limit_incremental_iterations
        self.limit_recursive_automatic = limit_recursive_automatic
        self.stratify = stratify
        self.verbose = verbose
        self.model_kwargs = model_kwargs

        self._explainer = ShapExplainerFactory.get_explainer(model=model)

        if automatic:
            assert (
                limit_automatic != None
            ), '"limit_automatic" must be specified when automatic mode is used!'

        # Log the column names for more interpretable column values
        self._input_names = None

    def _print(self, *values):
        """Helper method for printing if `verbose` is set to True."""
        if self.verbose:
            print(*values)

    def fit(self, X, y, stratify=None, cv=None):
        if stratify is None:
            stratify = self.stratify

        # Log the column names if X is a dataframe
        if isinstance(X, pd.DataFrame):
            self._input_names = X.columns.values

        # Perform the necessary sklearn checks -> X and y are both ndarray
        X, y = self._validate_data(X, y, multi_output=True)

        self._print("Starting PowerSHAP")

        X = pd.DataFrame(data=X, columns=list(range(X.shape[1])))

        loop_its = self.power_iterations
        if self.automatic:
            loop_its = 10
            self._print(
                "Automatic mode enabled: Finding the minimal required PowerSHAP",
                f"iterations for significance of {self.power_alpha}.",
            )

        shaps_df = self._explainer.explain(
            X=X,
            y=y,
            loop_its=loop_its,
            val_size=self.val_size,
            stratify=None,  # TODO: do this automatically?
        )
        processed_shaps_df = powerSHAP_statistical_analysis(
            shaps_df,
            self.power_alpha,
            self.power_req_iterations,
            include_all=self.include_all,
        )

        if self.automatic:
            max_iterations = int(
                np.ceil(
                    processed_shaps_df[processed_shaps_df.p_value < self.power_alpha][
                        str(self.power_req_iterations) + "_power_its_req"
                    ].max()
                )
            )

            max_iterations_old = loop_its
            recurs_counter = 0

            if max_iterations < max_iterations_old:
                self._print(
                    f"{loop_its} iterations were already sufficient as only",
                    f"{max_iterations} iterations were required for the current",
                    f"power_alpha = {self.power_alpha}.",
                )

            while (
                max_iterations > max_iterations_old
                # and max_iterations < limit_automatic
                and recurs_counter < self.limit_recursive_automatic
            ):

                shaps_df_recursive: pd.DataFrame = None
                if max_iterations > self.limit_automatic:
                    self._print(
                        f"Automatic mode: PowerSHAP Requires {max_iterations}",
                        "iterations; The required iterations exceed the limit_automatic",
                        "threshold. PowerSHAP will add",
                        f"{self.limit_incremental_iterations} PowerSHAP iterations and",
                        "re-evaluate.",
                    )

                    shaps_df_recursive = self._explainer.explain(
                        X=X,
                        y=y,
                        loop_its=self.limit_incremental_iterations,
                        val_size=self.val_size,
                        stratify=None,  # TODO: do this automatically?
                        random_seed_start=max_iterations_old,
                    )

                    max_iterations_old = (
                        max_iterations_old + self.limit_incremental_iterations
                    )

                else:
                    self._print(
                        f"Automatic mode: PowerSHAP Requires {max_iterations}"
                        f"iterations; Adding {max_iterations-max_iterations_old}",
                        "PowerSHAP iterations.",
                    )

                    shaps_df_recursive = self._explainer.explain(
                        X=X,
                        y=y,
                        loop_its=max_iterations - max_iterations_old,
                        val_size=self.val_size,
                        stratify=None,  # TODO: do this automatically?
                        random_seed_start=max_iterations_old,
                    )

                    max_iterations_old = max_iterations

                shaps_df = shaps_df.append(shaps_df_recursive)

                processed_shaps_df = powerSHAP_statistical_analysis(
                    shaps_df,
                    self.power_alpha,
                    self.power_req_iterations,
                    include_all=self.include_all,
                )

                max_iterations = int(
                    np.ceil(
                        processed_shaps_df[
                            processed_shaps_df.p_value < self.power_alpha
                        ][str(self.power_req_iterations) + "_power_its_req"].max()
                    )
                )

                recurs_counter = recurs_counter + 1

        self._print("Done!")

        self._processed_shaps_df = processed_shaps_df.copy()
        if self._input_names is not None:
            self._processed_shaps_df.index = [
                self._input_names[i] if isinstance(i, int) else i
                for i in processed_shaps_df.index.values
            ]

        # Remove the random feature (legit features have int index)
        sub_df = processed_shaps_df[
            processed_shaps_df.index.map(lambda x: isinstance(x, int))
        ]
        # Sort to have original order again
        sub_df = sub_df.sort_index()
        self._p_values = sub_df.p_value.values

        # It is convention to return self
        return self

    # This is the only method that needs to be implemented to serve the transform
    # functionality
    def _get_support_mask(self):
        # Select the significant features
        return self._p_values < self.power_alpha

    def transform(self, X):
        check_is_fitted(self, ["_processed_shaps_df", "_p_values"])
        if self._input_names is not None and isinstance(X, pd.DataFrame):
            assert np.all(X.columns.values == self._input_names)
            return pd.DataFrame(
                super().transform(X),
                columns=self._input_names[self._get_support_mask()],
            )
        return super().transform(X)
