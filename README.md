	
<p align="center">
    <a href="#readme">
        <img alt="PowerShap logo" src="https://raw.githubusercontent.com/predict-idlab/powershap/main/powershap_full_scaled.png" width=70%>
    </a>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/powershap.svg)](https://pypi.org/project/powershap/)
[![support-version](https://img.shields.io/pypi/pyversions/powershap)](https://img.shields.io/pypi/pyversions/powershap)
[![codecov](https://img.shields.io/codecov/c/github/predict-idlab/powershap?logo=codecov)](https://codecov.io/gh/predict-idlab/powershap)
[![Code quality](https://img.shields.io/lgtm/grade/python/github/predict-idlab/powershap?label=code%20quality&logo=lgtm)](https://lgtm.com/projects/g/predict-idlab/powershap/context:python)
[![Downloads](https://pepy.tech/badge/powershap)](https://pepy.tech/project/powershap)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?)](http://makeapullrequest.com)
[![Testing](https://github.com/predict-idlab/powershap/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/powershap/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/470633431.svg)](https://zenodo.org/badge/latestdoi/470633431)

> *powershap* is a **feature selection method** that uses statistical hypothesis testing and power calculations on **Shapley values**, enabling fast and intuitive wrapper-based feature selection.  

## Installation ⚙️

| [**pip**](https://pypi.org/project/powershap/) | `pip install powershap` | 
| ---| ----|

## Usage 🛠

*powershap* is built to be intuitive, it supports various models including linear, tree-based, and even deep learning models.  
<!-- It is also implented as sklearn `Transformer` component, allowing convenient integration in `sklearn` pipelines. -->

```py
from powershap import PowerShap
from catboost import CatBoostClassifier

X, y = ...  # your classification dataset

selector = PowerShap(
    model=CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=True)
)

selector.fit(X, y)  # Fit the PowerShap feature selector
selector.transform(X)  # Reduce the dataset to the selected features

```

## Features ✨

* default automatic mode
* `scikit-learn` compatible
* supports various models
* insights into the feature selection method: call the `._processed_shaps_df` on a fitted `PowerSHAP` feature selector.
* tested code!

## Benchmarks ⏱

Check out our benchmark results [here](examples/results/).  

## How does it work ⁉️

Powershap is built on the core assumption that *an informative feature will have a larger impact on the prediction compared to a known random feature.*

* Powershap trains multiple models with different random seeds on different subsets of the data. Each iteration it adds a random uniform feature to the dataset for training.
* In a single iteration after training a model, powershap calculates the absolute Shapley values of all features, including the random feature. If there are multiple outputs or multiple classes, powershap uses the maximum across these multiple outputs. These values are then averaged for each feature, symbolising the impact of the feature in this iteration.
* After performing all iterations, each feature then has an array of impacts. The impact array of each feature is then compared to the average of the random feature impact array using the percentile formula to provide a p-value. This tests whether the feature has a larger impact than the random feature and outputs a low p-value if true. 
* Powershap then outputs all features with a p-value below the provided threshold. The threshold is by default 0.01.


### Automatic mode 🤖

The required number of iterations and the threshold values are hyperparameters of powershap. However, to *avoid manually optimizing the hyperparameters* powershap by default uses an automatic mode that automatically determines these hyperparameters. 

* The automatic mode first starts with executing powershap using ten iterations.
* Then, for each feature powershap calculates the effect size and the statistical power of the test using a student-t power test. 
* Using the calculated effect size, powershap then calculates the required iterations to achieve a predefined power requirement. By default this is 0.99, which represents a false positive probability of 0.01.
* If the required iterations are larger than the already performed iterations, powershap then further executes for the extra required iterations. 
* Afterward, powershap re-calculates the required iterations and it keeps re-executing until the required iterations are met.

## Referencing our package :memo:

If you use *powershap* in a scientific publication, we would highly appreciate citing us as:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.08394,
  doi = {10.48550/ARXIV.2206.08394},
  url = {https://arxiv.org/abs/2206.08394},
  author = {Verhaeghe, Jarne and Van Der Donckt, Jeroen and Ongenae, Femke and Van Hoecke, Sofie},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Powershap: A Power-full Shapley Feature Selection Method},
  publisher = {arXiv},
  year = {2022}
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

Paper is accepted at ECML PKDD 2022 and will be presented there. The preprint can be found on arXiv ([link](https://arxiv.org/abs/2206.08394)) and on the github.

---

<p align="center">
👤 <i>Jarne Verhaeghe, Jeroen Van Der Donckt</i>
</p>
