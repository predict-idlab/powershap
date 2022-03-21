import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm
from numpy.random import RandomState
from PowerSHAP.Base import base_functions



def catBoostSHAP(
    current_df,
    model,
    target_column,
    feature_columns_random=None,
    index_column=None,
    loop_its=10,
    split_data=True,
    val_size=0.2,
    stratify=None,
    random_seed_start = 0
):
    Shaps = np.array([])

    for i in tqdm(range(loop_its)):
        npRandomState = RandomState(i+random_seed_start)

        random_uniform_feature = npRandomState.uniform(-1, 1, len(current_df))
        current_df["random_uniform_feature"] = random_uniform_feature

        if split_data:
            train_idx, val_idx = train_test_split(
                current_df[index_column].unique(),
                test_size=val_size,
                random_state=i,
                stratify=stratify,
            )
        else:
            train_idx = current_df[index_column].unique()
            val_idx = current_df[index_column].unique()

        X_train = current_df[current_df[index_column].isin(train_idx)].copy(deep=True)
        X_val = current_df[current_df[index_column].isin(val_idx)].copy(deep=True)

        ## Calculate the required labels

        Y_train = X_train[target_column].values
        Y_val = X_val[target_column].values

        ## Extract the required features
        X_train_feat = X_train[feature_columns_random]
        X_val_feat = X_val[feature_columns_random]

        PowerSHAP_model = model.copy().set_params(random_seed=i+random_seed_start)
        PowerSHAP_model.fit(X_train_feat, Y_train, eval_set=(X_val_feat, Y_val))

        C_explainer = shap.TreeExplainer(PowerSHAP_model)

        Shap_values = C_explainer.shap_values(X_val_feat)

        Shap_values = np.abs(Shap_values)

        if len(Shaps) > 0:
            Shaps = np.vstack([Shaps, np.mean(Shap_values, axis=0)])
        else:
            Shaps = np.mean(Shap_values, axis=0)

    return pd.DataFrame(data=Shaps, columns=feature_columns_random)



# input_df: Input DataFrame for training (Pandas Dataframe)
# feature_columns: The feature columns used in the input_df DataFrame (List)
# powershap_iterations: Indicating the amount of shuffles and iterations of the method, ignored when using automatic mode (int)
# index_column: The index column of the input_df Dataframe, this is used to also split the database into a train and validation set (String)
# target_column: The target column of the input_df Dataframe (String)
# model: The model used for the iteration (CatBoost model)
# val_size: The fractional validation set size (float between [0,1])
# stratify: Stratify the train_test_split (Boolean)
# include_all: If false, this method will output all features with a threshold of power_alpha (Boolean)
# power_alpha: The alpha value used for the power-calculation of the used statistical test and significance threshold (float between ]0,1[)
# power_req_iterations: The fractional power percentage for the required iterations calculation
# split_data: If false, the data will not be split into a validation set and Powershap will only use the training data
# automatic: If true, the PowerSHAP will first calculate the required iterations by using ten iterations and then restart using the required iterations for power_alpha (Boolean)
# limit_automatic: sets a limit to the maximum allowed iterations (int)
# limit_incremental_iterations: if the required iterations exceed limit_automatic in automatic mode, add limit_incremental_iterations iterations and re-evaluate. 
# limit_recursive_automatic: restricts the amount of automatic PowerSHAP recursion (int)
def PowerSHAP(
    input_df,
    model,
    target_column,
    feature_columns=None,
    index_column=None,
    powershap_iterations=10,
    val_size=0.2,
    stratify=None,
    power_alpha=0.01,
    include_all=False,
    power_req_iterations=0.95,
    split_data=True,
    automatic=False,
    limit_automatic=None,
    limit_incremental_iterations=10,
    limit_recursive_automatic=3,
):
    print("Starting PowerSHAP")

    #Shaps = np.array([])
    # Shap_values_ar = np.array([])
    # X_Shap_ar = np.array([])

    current_df = input_df.copy(deep=True)

    if index_column == None:
        current_df = current_df.reset_index()
        index_column = "index"

    if feature_columns == None:
        feature_columns = list(current_df.drop(columns=[target_column,index_column]).columns.values)

    columns_list_current = feature_columns + ["random_uniform_feature"]

    if automatic:
        assert (
            limit_automatic != None
        ), "\"limit_automatic\" must be specified when automatic mode is used"
        loop_its = 10
        print(
            "Automatic mode enabled: Finding the minimal required PowerSHAP iterations for significance of "
            + str(power_alpha)
            + "."
        )
    else:
        loop_its = powershap_iterations

    shaps_df = catBoostSHAP(
        current_df,
        model,
        target_column,
        feature_columns_random=columns_list_current,
        index_column=index_column,
        loop_its=loop_its,
        split_data = split_data,
        val_size=val_size,
        stratify=stratify,
        random_seed_start = 0
    )

    processed_shaps_df = base_functions.powerSHAP_statistical_analysis(shaps_df,power_alpha,power_req_iterations,include_all)

    if automatic:
        max_iterations = int(
            np.ceil(
                processed_shaps_df[processed_shaps_df.p_value < power_alpha][
                    str(power_req_iterations)+"_power_its_req"
                ].max()
            )
        )

        max_iterations_old = loop_its
        recurs_counter = 0

        if max_iterations <= max_iterations_old:
            print(
                str(loop_its)+" iterations were already sufficient as only " 
                + str(max_iterations) 
                + " iterations were required for the current power_alpha = "+ str(power_alpha) + "."
            )

        while (
                max_iterations > max_iterations_old
                #and max_iterations < limit_automatic
                and recurs_counter < limit_recursive_automatic
            ):

            if max_iterations > limit_automatic:
                print(
                    "Automatic mode: PowerSHAP Requires " 
                    + str(max_iterations) + " iterations; "
                    + "The required iterations exceed the limit_automatic threshold. PowerSHAP will add "
                    + str(limit_incremental_iterations) + " PowerSHAP iterations and re-evaluate."
                )

                shaps_df_recursive = catBoostSHAP(
                    current_df,
                    model,
                    target_column,
                    feature_columns_random=columns_list_current,
                    index_column=index_column,
                    loop_its=limit_incremental_iterations,
                    val_size=val_size,
                    stratify=stratify,
                    random_seed_start = max_iterations_old
                )

                max_iterations_old = max_iterations_old + limit_incremental_iterations

            else:
                print(
                    "Automatic mode: PowerSHAP Requires "+str(max_iterations) + " iterations; Adding "
                    + str(max_iterations-max_iterations_old)
                    + " PowerSHAP iterations."
                )

                if max_iterations-max_iterations_old==1:
                    max_iterations = max_iterations + 1
                    print("Adding another iteration to have at least two iterations.")

                shaps_df_recursive = catBoostSHAP(
                    current_df,
                    model,
                    target_column,
                    feature_columns_random=columns_list_current,
                    index_column=index_column,
                    loop_its=max_iterations-max_iterations_old,
                    val_size=val_size,
                    stratify=stratify,
                    random_seed_start = max_iterations_old
                )

                max_iterations_old = max_iterations

            shaps_df = shaps_df.append(shaps_df_recursive)

            processed_shaps_df = base_functions.powerSHAP_statistical_analysis(shaps_df,power_alpha,power_req_iterations,include_all)
            
            max_iterations = int(
                np.ceil(
                    processed_shaps_df[processed_shaps_df.p_value < power_alpha][
                        str(power_req_iterations)+"_power_its_req"
                    ].max()
                )
            )

            recurs_counter = recurs_counter+1

    print("Done!")
    if include_all:
        return processed_shaps_df  # ,shaps_df,Shap_values_ar,X_Shap_ar

    else:
        return processed_shaps_df[
            processed_shaps_df.p_value < power_alpha
        ]  # ,shaps_df,Shap_values_ar,X_Shap_ar
