# PowerSHAP

> *powershap* is a **feature selection method** that uses statistical hypothesis testing and power calculations on **Shapley values**, enabling fast and intuitive wrapper-based feature selection.  

## Installation

| [**pip**](https://pypi.org/project/powershap/) | `pip install powershap` | 
| ---| ----|

## Usage

*powershap* is built to be intuitive, it supports various models including linear, tree-based, and even deep learning models.  
<!-- It is also implented as sklearn `Transformer` component, allowing convenient integration in `sklearn` pipelines. -->

```py
from powershap import PowerSHAP
from catboost import CatboostClassifier

X, y = ...  # your classification dataset

selector = PowerSHAP(
    model=CatboostClassifier(n_estimators=250, verbose=0)
)

selector.fit(X, y)  # Fit the PowerSHAP feature selector
selector.transform(X)  # Reduce the dataset to the selected features

```

## Features

* default automatic mode
* `scikit-learn` compatible
* supports various models
* insights into the feature selection method: call the `._processed_df` on a fitted `PowerSHAP` feature selector.
* tested code!

## Benchmarks

Check out our benchmark results [here](examples/results/).  

## How it works

Powershap is built on the core assumption that *an informative feature will have a larger impact on the prediction compared to a known random feature.*

...

### Automatic mode

...

---

