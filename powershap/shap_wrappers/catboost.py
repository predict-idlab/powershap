__author__ = "Jarne Verheaghe"

import shap
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, CatBoostRegressor


def catBoostSHAP(
    model,
    X: pd.DataFrame,  #     current_df,
    y: np.array,  #    target_column,
    # feature_columns_random=None,
    # index_column=None,
    loop_its: int,
    val_size: float,
    stratify=None,
    random_seed_start=0,
) -> pd.DataFrame:
    assert not "random_uniform_feature" in X.columns
    assert isinstance(model, (CatBoostClassifier, CatBoostRegressor))

    shaps = np.array([])  # TODO: pre-allocate for efficiency

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
        X_train = X.iloc[train_idx].copy(deep=True)
        X_val = X.iloc[val_idx].copy(deep=True)
        Y_train = y[train_idx]
        Y_val = y[val_idx]

        # Fit the model
        PowerSHAP_model = model.copy().set_params(random_seed=i + random_seed_start)
        PowerSHAP_model.fit(X_train, Y_train, eval_set=(X_val, Y_val))

        # Calculate the shap values
        C_explainer = shap.TreeExplainer(PowerSHAP_model)
        Shap_values = C_explainer.shap_values(X_val)

        Shap_values = np.abs(Shap_values)

        # TODO: np.concatenate on a list
        # TODO: consider to convert to even float16?
        if len(shaps) > 0:
            shaps = np.vstack([shaps, np.mean(Shap_values, axis=0).astype("float32")])
        else:
            shaps = np.mean(Shap_values, axis=0).astype("float32")

    return pd.DataFrame(data=shaps, columns=X_train.columns.values)
