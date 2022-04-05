__author__ = "Jarne Verhaeghe"

import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.power import TTestPower

def p_values_arg_coef(coefficients, arg):
    return stats.percentileofscore(coefficients, arg)


def powerSHAP_statistical_analysis(
    shaps_df: pd.DataFrame,
    power_alpha: float,
    power_req_iterations: float,
    include_all: bool,
):
    p_values = []
    effect_size = []
    power_list = []
    required_iterations = []
    n_samples = len(shaps_df["random_uniform_feature"].values)
    mean_random_uniform = shaps_df["random_uniform_feature"].mean()
    for i in range(len(shaps_df.columns)):
        p_value = (
            p_values_arg_coef(
                np.array(shaps_df.values[:, i]), mean_random_uniform)
            / 100
        )

        p_values.append(p_value)

        if include_all or p_value < power_alpha:
            pooled_standard_deviation = np.sqrt(
                (
                    (shaps_df.std().values[i] ** 2) 
                    + (shaps_df["random_uniform_feature"].values.std() ** 2)
                )
                / (2)
            )
            effect_size.append(
                (mean_random_uniform - shaps_df.mean().values[i])
                / pooled_standard_deviation
            )
            power_list.append(
                TTestPower().power(
                    effect_size=effect_size[-1],
                    nobs=n_samples,
                    alpha=power_alpha,
                    df=None,
                    alternative="smaller",
                )
            )
            if shaps_df.columns[i] == "random_uniform_feature":
                required_iterations.append(0)
            else:
                required_iterations.append(
                    TTestPower().solve_power(
                        effect_size=effect_size[-1],
                        nobs=None,
                        alpha=power_alpha,
                        power=power_req_iterations,
                        alternative="smaller",
                    )
                )

        else:
            required_iterations.append(0)
            effect_size.append(0)
            power_list.append(0)

    processed_shaps_df = pd.DataFrame(
        data=np.hstack(
            [
                np.reshape(shaps_df.mean().values, (-1, 1)),
                np.reshape(np.array(p_values), (len(p_values), 1)),
                np.reshape(np.array(effect_size), (len(effect_size), 1)),
                np.reshape(np.array(power_list), (len(power_list), 1)),
                np.reshape(
                    np.array(required_iterations), (len(required_iterations), 1)
                ),
            ]
        ),
        columns=[
            "impact",
            "p_value",
            "effect_size",
            "power_" + str(power_alpha) + "_alpha",
            str(power_req_iterations) + "_power_its_req",
        ],
        index=shaps_df.mean().index,
    )
    processed_shaps_df = processed_shaps_df.reindex(
        processed_shaps_df.impact.abs().sort_values(ascending=False).index
    )

    return processed_shaps_df
