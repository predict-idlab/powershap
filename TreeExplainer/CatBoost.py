import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm
from numpy.random import RandomState
from PowerSHAP.Base import base_functions
from statsmodels.stats.power import TTestPower


########################################################################################################################
# Inp_db: Input DataFrame for training (Pandas Dataframe)
# Feature_cols: The feature columns used in the Inp_db DataFrame (List)
# powershap_iterations: Indicating the amount of shuffles and iterations of the method, ignored when using automatic mode (int)
# Index: The index column of the Inp_db Dataframe, this is used to also split the database into a train and validation set (String)
# model: The model used for the iteration (CatBoost model)
# val_size: The fractional validation set size (float between [0,1])
# stratify: Stratify the train_test_split (Boolean)
# include_all: If false, this method will result all features with a threshold of alpha and not save all shap_values and corresponding data samples (Boolean)
# power_alpha: The alpha value used for the power-calculation of the used statistical test and significance threshold (float between ]0,1[)
# power_req_iterations: The fractional power percentage for the required iterations calculation
# automatic: If true, the PowerSHAP will first calculate the required iterations by using ten iterations and then restart using the required iterations for power_alpha (Boolean)
# limit_recursive_automatic: restricts the amount of automatic PowerSHAP recursion (int)
# limit_automatic: sets a limit to the maximum allowed iterations (int)
def PowerSHAP(Inp_db, Feature_cols, Target, Index, 
    model,powershap_iterations=10, val_size=0.2,stratify=None,power_alpha=0.01,include_all=False,
    power_req_iterations=0.95,automatic=False,limit_automatic=None,limit_recursive_automatic=3):
    print("Starting PowerSHAP")

    Shaps = np.array([])
    Shap_values_ar = np.array([])
    X_Shap_ar = np.array([])
    
    current_df = Inp_db.copy(deep=True)

    columns_list_current = Feature_cols+["random_uniform_feature"]

    if automatic:
        assert limit_automatic!= None, "limit_automatic must be specified when automatic mode is used"
        loop_its = 10
        print("Automatic mode enabled: Finding the minimal required PowerSHAP iterations for significance of "+str(power_alpha)+".")
    else:
        loop_its = powershap_iterations

    for i in tqdm(range(loop_its)):
        npRandomState = RandomState(i)

        random_uniform_feature = npRandomState.uniform(-1,1,len(current_df))
        current_df["random_uniform_feature"]=random_uniform_feature

        train_idx,val_idx = train_test_split(current_df[Index].unique(),test_size=val_size,random_state = i,stratify=stratify)

        X_train = current_df[current_df[Index].isin(train_idx)].copy(deep=True)
        X_val = current_df[current_df[Index].isin(val_idx)].copy(deep=True)

        ## Calculate the required labels

        Y_train = X_train[Target].values
        Y_val = X_val[Target].values

        ## Extract the required features
        X_train_feat = X_train[columns_list_current]
        X_val_feat = X_val[columns_list_current]

        PowerSHAP_model = model.copy().set_params(random_seed=i)
        PowerSHAP_model.fit(X_train_feat,Y_train,eval_set=(X_val_feat,Y_val))

        C_explainer = shap.TreeExplainer(PowerSHAP_model)

        Shap_values = C_explainer.shap_values(X_val_feat)

        if include_all:
            if len(Shap_values_ar)>0:
                Shap_values_ar = np.vstack([Shap_values_ar,Shap_values])
                X_Shap_ar = np.vstack([X_Shap_ar,X_val_feat])
            else:
                Shap_values_ar = Shap_values
                X_Shap_ar = X_val_feat

        Shap_values = np.abs(Shap_values)

        if len(Shaps)>0:
            Shaps = np.vstack([Shaps,np.mean(Shap_values,axis=0)])
        else:
            Shaps = np.mean(Shap_values,axis=0)

    shaps_pd = pd.DataFrame(data=Shaps,columns=columns_list_current)

    ############################################################################
    ## SHAP STATISTICAL ANALYSIS
    p_values = []
    effect_size = []
    power_list = []
    required_iterations = []
    n_samples = len(shaps_pd["random_uniform_feature"].values)
    mean_random_uniform = shaps_pd["random_uniform_feature"].values.mean()
    for i in range(len(shaps_pd.columns)):
        quantile = base_functions.p_values_arg_coef(np.array(shaps_pd.values[:,i]),mean_random_uniform)/100
        p_value = quantile
        p_values.append(p_value)

        if include_all or p_value < power_alpha:
            pooled_standard_deviation = np.sqrt(((shaps_pd.std()[i]**2)*(n_samples-1)+(n_samples-1)*(shaps_pd["random_uniform_feature"].values.std()**2))/(n_samples*2-2))
            effect_size.append((np.abs(shaps_pd.mean()[i]-mean_random_uniform))/pooled_standard_deviation)
            power_list.append(TTestPower().power(effect_size=effect_size[-1],nobs=n_samples,alpha=power_alpha,df=None,alternative="larger"))
            if shaps_pd.columns[i]=="random_uniform_feature":
                required_iterations.append(0)
            else:
                required_iterations.append(TTestPower().solve_power(effect_size=effect_size[-1],nobs=None,alpha=power_alpha,power=power_req_iterations,alternative="larger"))

        else:
            required_iterations.append(0)
            effect_size.append(0)
            power_list.append(0)

    processed_shaps_pd = pd.DataFrame(data=np.hstack([np.reshape(shaps_pd.mean().values,(-1,1)),
        np.reshape(np.array(p_values),(len(p_values),1)),
        np.reshape(np.array(effect_size),(len(effect_size),1)),
        np.reshape(np.array(power_list),(len(power_list),1)),
        np.reshape(np.array(required_iterations),(len(required_iterations),1))]),columns=["Impact","p_value","effect_size","power_0_01_alpha","95_power_its_req"],index=shaps_pd.mean().index)
    processed_shaps_pd = processed_shaps_pd.reindex(processed_shaps_pd.Impact.abs().sort_values(ascending=False).index)
    
    if automatic:
        max_iterations = int(np.ceil(processed_shaps_pd[processed_shaps_pd.p_value<0.01]["95_power_its_req"].max()))

        if max_iterations > limit_automatic:
            print("The required iterations exceed the limit_automatic threshold. Powershop will continue with only "+str(limit_automatic)+" iterations.")
            processed_shaps_pd = PowerSHAP(Inp_db=Inp_db, Feature_cols=Feature_cols, Target=Target, Index=Index, 
                                            model=model,powershap_iterations=limit_automatic, val_size=val_size,stratify=stratify,power_alpha=power_alpha,include_all=True,
                                            power_req_iterations=power_req_iterations,automatic=False)
        else: 
            max_iterations_old = 1
            recurs_counter = 0
            while max_iterations > max_iterations_old and max_iterations < limit_automatic and recurs_counter < limit_recursive_automatic:
                print("Automatic mode: Requires more iterations; Restarting PowerSHAP with "+str(max_iterations)+" iterations.")
                processed_shaps_pd = PowerSHAP(Inp_db=Inp_db, Feature_cols=Feature_cols, Target=Target, Index=Index, 
                                            model=model,powershap_iterations=max_iterations, val_size=val_size,stratify=stratify,power_alpha=power_alpha,include_all=True,
                                            power_req_iterations=power_req_iterations,automatic=False)

                max_iterations_old = max_iterations
                max_iterations = int(np.ceil(processed_shaps_pd[processed_shaps_pd.p_value<0.01]["95_power_its_req"].max()))
                recurs_counter = recurs_counter+1

    print("Done!")
    if include_all:
        return processed_shaps_pd#,shaps_pd,Shap_values_ar,X_Shap_ar

    else:
        return processed_shaps_pd[processed_shaps_pd.p_value<power_alpha]#,shaps_pd,Shap_values_ar,X_Shap_ar


